import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm, trange

from src.data.mnist import get_mnist_loaders

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, H, W = self.weight.size()
        self.mask.fill_(1)
        # Mask out future pixels (below and to the right)
        self.mask[:, :, H // 2, W // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, H // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNN(L.LightningModule):
    def __init__(self, in_channels=1, hidden_size=64, num_layers=7, condition_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.condition_size = condition_size
        
        # Initial Mask A layer (cannot see the current pixel)
        self.layers = nn.ModuleList([
            MaskedConv2d('A', in_channels, hidden_size, 7, padding=3)
        ])
        
        # Label conditioning projection
        self.label_proj = nn.Linear(condition_size, hidden_size)

        # Subsequent Mask B layers (can see previous features and current pixel)
        for _ in range(num_layers - 2):
            self.layers.append(MaskedConv2d('B', hidden_size, hidden_size, 7, padding=3))
        
        # Final layer to map to 256 intensity levels (discretized pixels)
        self.layers.append(MaskedConv2d('B', hidden_size, 256, 1))

    def forward(self, x, y):
        h = x
        label_emb = self.label_proj(y.float()) 
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                # Inject conditioning into every layer
                h = h + label_emb[:, :, None, None]
                h = F.relu(h)
        return h

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Map [0, 1] float to [0, 255] long for CrossEntropy
        target = (x * 255).long().squeeze(1) 
        logits = self(x, y)
        loss = F.cross_entropy(logits, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Map [0, 1] float to [0, 255] long for CrossEntropy
        target = (x * 255).long().squeeze(1) 
        logits = self(x, y)
        loss = F.cross_entropy(logits, target)
        self.log("val_loss", loss)


    @torch.no_grad()
    def sample(self, n_samples: int, device=None, batch_size=256, **kwargs):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device=device)
            
        labels = torch.arange(10).repeat_interleave(n_samples).to(device)
        y_all = F.one_hot(labels, num_classes=self.condition_size).float()
        
        n_total = n_samples * 10
        samples_all = []

        for i in trange(0, n_total, batch_size, desc=f"Generating {n_samples*10} samples...", leave=False, disable=not sys.stdout.isatty()):
            curr_batch_size = min(batch_size, n_total - i)
            y = y_all[i : i + curr_batch_size]
            
            canvas = torch.zeros(curr_batch_size, 1, 28, 28).to(device)

            for row in range(28):
                for col in range(28):
                    logits = self(canvas, y)
                    probs = F.softmax(logits[:, :, row, col], dim=1)
                    pixel_val = torch.multinomial(probs, 1)
                    canvas[:, :, row, col] = pixel_val.float() / 255.0
            
            samples_all.append(canvas)
            
        return torch.cat(samples_all, dim=0), y_all

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def get_trained(train=True, max_epochs=10, **kwargs):
    model = PixelCNN()
    chkpt_path = Path("checkpoints/pixelcnn_mnist.ckpt")
    
    if train:
        trl, tel = get_mnist_loaders()
        logger = TensorBoardLogger(os.getcwd()+"/trained_models", name="pixelcnn_logs")
        trainer = L.Trainer(max_epochs=max_epochs, logger=logger, accelerator="auto")
        trainer.fit(model, trl)
        trainer.save_checkpoint(chkpt_path)
    else:
        if chkpt_path.exists():
            model = PixelCNN.load_from_checkpoint(chkpt_path)
        else:
            print("No checkpoint found. Training instead...")
            return get_trained(train=True, max_epochs=max_epochs, **kwargs)
            
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    
    # 1. Train or Load
    m = get_trained(train=args.train, max_epochs=args.max_epochs)
    m.eval()
    m = torch.compile(m, mode='reduce-overhead')

    # 2. Sample
    with torch.no_grad():
        print("Generating 600 samples...")
        s, y = m.sample(n_samples=60, batch_size=128)
        print(f"Sample shape: {s.shape}, Labels shape: {y.shape}")

    # 3. Evaluate using your existing CNN classifier pipeline
    try:
        from src.models.classification.cnn import get_trained as get_classifier
        from src.metrics.prediction_accuracy import compute_metrics
        from src.metrics.entropy_metrics import compute_entropy_metrics
        
        print("Evaluating sample quality via classifier...")
        classifier = get_classifier()
        trainer = L.Trainer(accelerator="auto")
        
        # Predict labels for the generated images
        pred_data = DataLoader(TensorDataset(s, y), batch_size=128)
        predictions = trainer.predict(classifier, dataloaders=pred_data)
        probslist, labelslist, _ = zip(*predictions)

        y_pred_probs = torch.cat(probslist)
        
        # Compute accuracy (how often the classifier agrees with what the model *tried* to generate)
        acc, cm = compute_metrics(y.argmax(dim=-1).cpu(), y_pred_probs.cpu())
        print(f"\nModel Realism Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix:\n", cm)
        
    except ImportError:
        print("Classifier or metrics modules not found. Skipping evaluation.")