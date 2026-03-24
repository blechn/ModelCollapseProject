from argparse import ArgumentParser
import os
from pathlib import Path

from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn, optim
import torch.nn.functional as F

import lightning as L

from src.data.mnist import get_mnist_loaders


class CNNClassifierModule(nn.Module):
    def __init__(self, dropout_prob: float = 0.3, **kwargs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # after this, shape is (B, 64, 28, 28)
            nn.Dropout(p=dropout_prob),
            nn.Flatten(start_dim=1, end_dim=-1),  # 64 * 28 * 28
            nn.Linear(in_features=64 * 28 * 28, out_features=1024),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class CNNClassifier(L.LightningModule):
    def __init__(self, model: CNNClassifierModule = None, **kwargs):
        super().__init__()
        if model is None:
            model = CNNClassifierModule(**kwargs)
            model.to(device=kwargs.get('device', torch.device('cuda:0')))
            model = torch.compile(model, fullgraph=True, mode='reduce-overhead')
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        return loss

    def configure_optimizers(self, **kwargs):
        optimizer = optim.Adam(self.parameters(), lr=kwargs.get('lr', 1e-3))
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss)

    def forward(self, x):
        return self.model(x)

    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1)

    def predict_label(self, x):
        probs = self.predict_prob(x)
        return torch.argmax(probs, dim=1)

    def predict(self, x):
        probs = self.predict_prob(x)
        labels = self.predict_label(x)
        return probs, labels

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        probs, labels = self.predict(x)
        features = self.get_features(x)
        return probs, labels, features

    def get_features(self, x):
        features = self.model.layers[:-2](x) # get the features in 64 dimensional space for a nice representation
        return features

def get_trained(**kwargs):
    L.pytorch.utilities.disable_possible_user_warnings()
    path = kwargs.get('path', Path(os.getcwd()+'/trained_models'))
    if kwargs.get('train', False):
        print("Training new classifier...")
        model = CNNClassifier()
        logger_add = 'cnnclassifier_logs'
        logger_add += '_fashion_';print("Using FashionMNIST") if kwargs.get('fashion') else ''
        trainer = L.Trainer(logger=TensorBoardLogger(path, 'cnnclassifier_logs'), max_epochs=kwargs.get('max_epochs', 20), accelerator='cuda')

        trl, tel = get_mnist_loaders(**kwargs)

        trainer.fit(model=model, train_dataloaders=trl, val_dataloaders=tel)
    else:
        if kwargs.get('fashion'):
            model = CNNClassifier.load_from_checkpoint(path/Path('cnnclassifier_logs/version_15/checkpoints/epoch=4-step=4690.ckpt')) # hardcoded, good model for FashionMNIST
        else:
            model = CNNClassifier.load_from_checkpoint(path/Path('cnnclassifier_logs/version_5/checkpoints/epoch=4-step=4690.ckpt')) # hardcoded, good model for MNIST



    print("Model loaded.")
    
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False, help="Whether to train a model or just load it from the latest checkpoint.")
    parser.add_argument("--max_epochs", action="store", type=int, default=4, help="How many epochs to train the CNN.")
    parser.add_argument("--fashion", action="store_true", default=False, help="Use the FashionMNIST dataset instead of the original MNIST.")
    torch.set_float32_matmul_precision('high')
    args = parser.parse_args()
    args_dict = vars(args)

    m = get_trained(**args_dict).to('cuda')
    # predict test
    dummy_x = torch.randn((5000, 1, 28, 28)).to('cuda')
    dummy_y = torch.randint(0, 10, (5000,)).to('cuda')
    dummy_y_onehot = torch.nn.functional.one_hot(dummy_y)

    with torch.no_grad():
        p, l = m.predict(dummy_x)
        print(p.shape, l.shape, f.shape)

    from src.metrics.prediction_accuracy import compute_metrics
    acc, cm = compute_metrics(dummy_y, p)
    print(f"Accuracy: {acc}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(cm.cpu(), cmap='Blues', annot=True, fmt='d')
    plt.show()
    plt.close()