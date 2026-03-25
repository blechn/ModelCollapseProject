import os
import sys
import math
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torchdiffeq import odeint

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from tqdm import trange

from src.data.mnist import get_mnist_loaders


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        attn_out, _ = self.mha(h, h, h)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return x + attn_out

class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(x))
        emb_out = self.emb_proj(self.act(emb))[:, :, None, None]
        h = h + emb_out
        h = self.conv2(self.act(h))
        return self.shortcut(x) + h

class VectorFieldNet(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, condition_size=10):
        super().__init__()
        self.condition_size = condition_size
        time_emb_dim = hidden_size * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_proj = nn.Linear(condition_size, time_emb_dim)

        self.init_conv = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        
        self.down1 = ConditionalResBlock(hidden_size, hidden_size, time_emb_dim)
        self.pool1 = nn.Conv2d(hidden_size, hidden_size, 4, stride=2, padding=1) # 28x28 -> 14x14
        
        self.down2 = ConditionalResBlock(hidden_size, hidden_size * 2, time_emb_dim)
        self.pool2 = nn.Conv2d(hidden_size * 2, hidden_size * 2, 4, stride=2, padding=1) # 14x14 -> 7x7
        
        self.down3 = ConditionalResBlock(hidden_size * 2, hidden_size * 2, time_emb_dim)

        self.mid1 = ConditionalResBlock(hidden_size * 2, hidden_size * 2, time_emb_dim)
        self.attn = SelfAttention(hidden_size * 2)
        self.mid2 = ConditionalResBlock(hidden_size * 2, hidden_size * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(hidden_size * 2, hidden_size * 2, 4, stride=2, padding=1) # 7x7 -> 14x14
        self.up_block1 = ConditionalResBlock(hidden_size * 4, hidden_size * 2, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, stride=2, padding=1) # 14x14 -> 28x28
        self.up_block2 = ConditionalResBlock(hidden_size * 2, hidden_size, time_emb_dim)

        self.final_conv = nn.Conv2d(hidden_size, in_channels, 3, padding=1)

    def forward(self, x_t, t, y):
        emb = self.time_mlp(t) + self.label_proj(y.float())

        h0 = self.init_conv(x_t)
        
        h1 = self.down1(h0, emb)
        p1 = self.pool1(h1)
        
        h2 = self.down2(p1, emb)
        p2 = self.pool2(h2)
        
        h3 = self.down3(p2, emb)
        
        m = self.mid2(self.attn(self.mid1(h3, emb)), emb)
        
        u1 = self.up1(m)
        ub1 = self.up_block1(torch.cat([u1, h2], dim=1), emb)
        
        u2 = self.up2(ub1)
        ub2 = self.up_block2(torch.cat([u2, h1], dim=1), emb)
        
        out = self.final_conv(ub2)
        return out


class FlowMatching(L.LightningModule):
    def __init__(self, model: VectorFieldNet | None = None, **kwargs):
        super().__init__()
        if model is None:
            model = VectorFieldNet(in_channels=1, hidden_size=64)
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def loss_fn(self, x1, y):
        x1 = x1 * 2.0 - 1.0

        B = x1.shape[0]

        t = torch.rand((B,), device=x1.device)

        x0 = torch.randn_like(x1)

        t_expand = t.view(B, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        u_t = x1 - x0

        v_t = self.model(x_t, t, y)

        return F.mse_loss(v_t, u_t)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(x, y)
        self.log("val_loss", loss)

    def configure_optimizers(self, **kwargs):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.get("lr", 1e-3))
        return optimizer

    @torch.no_grad()
    def sample(self, n_samples: int = 1, batch_size: int = 512, device=None, **kwargs):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.eval()
        condition_size = self.model.condition_size

        y_full = (
            torch.nn.functional.one_hot(
                torch.arange(0, condition_size, device=device).repeat_interleave(
                    n_samples
                )
            )
        ).float()

        total_to_generate = y_full.shape[0]
        decoded_samples = []

        def ode_func(t, x_t, y_batch):
            # t is a scalar tensor so we have to convert it to a batch
            t_batch = torch.full((x_t.shape[0],), t.item(), device=device)
            return self.model(x_t, t_batch, y_batch)

        for i in trange(
            0,
            total_to_generate,
            batch_size,
            desc="Generating data",
            leave=True,
            disable=not sys.stdout.isatty(),
        ):
            y_batch = y_full[i : i + batch_size]
            current_batch_size = y_batch.shape[0]

            x_0 = torch.randn((current_batch_size, 1, 28, 28), device=device)

            trajectory = odeint(
                lambda t, x: ode_func(t, x, y_batch),
                x_0,
                torch.tensor([0.0, 1.0], device=device),
                method="dopri5",
                atol=1e-2,
                rtol=1e-2,
            )

            x_1 = trajectory[-1]  # final step is the generated image

            # scale back from [-1, 1] to [0, 1] and move to cpu
            x_1 = (x_1 + 1.0) / 2.0
            x_1 = torch.clamp(x_1, 0.0, 1.0)
            decoded_samples.append(x_1.cpu())

        return torch.cat(decoded_samples, dim=0).to(device), y_full.to(
            device
        )  # create tensor from all generated batches


def get_trained(**kwargs):
    L.pytorch.utilities.disable_possible_user_warnings()
    path = kwargs.get("path", Path(os.getcwd() + "/trained_models"))

    if kwargs.get("train", False):
        print("Training new Flow Matching model...")
        model = FlowMatching()

        trainer = L.Trainer(
            logger=TensorBoardLogger(path, "flowmatch_logs"),
            max_epochs=kwargs.get("max_epochs", 20),
            accelerator="cuda",
        )

        trl, tel = get_mnist_loaders()

        trainer.fit(model=model, train_dataloaders=trl, val_dataloaders=tel)
    else:
        checkpoint_path = path / Path(
            "flowmatch_logs/version_1/checkpoints/epoch=19-step=18760.ckpt"  # hardcoded, small model
        )
        if not checkpoint_path.exists():
            print(
                "Checkpoint not found. Please train the model first by passing --train."
            )
            return None
        model = FlowMatching.load_from_checkpoint(checkpoint_path)

    print("Model loaded.")
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether to train a model or just load it.",
    )
    parser.add_argument(
        "--max_epochs",
        action="store",
        type=int,
        default=10,
        help="How many epochs to train.",
    )
    torch.set_float32_matmul_precision("high")
    args = parser.parse_args()
    args_dict = vars(args)

    m = get_trained(**args_dict)

    if m is not None:
        m.eval()
        with torch.no_grad():
            s, y = m.sample(n_samples=1000, steps=50)
            print("Sample shape:", s.shape, "Labels shape:", y.shape)

        # Standard classification checks, matching your pipeline
        from src.models.classification.cnn import get_trained as get_c

        c = get_c()
        t = L.Trainer()
        predictions = t.predict(
            c, dataloaders=DataLoader(TensorDataset(s, y), batch_size=128)
        )

        probslist, labelslist, _ = zip(*predictions)
        y_pred = torch.cat(probslist)
        y_pred2 = torch.cat(labelslist)

        from src.metrics.prediction_accuracy import compute_metrics as compute_acc_cm

        acc, cm = compute_acc_cm(y.argmax(dim=-1).cpu(), y_pred.cpu())
        acc2, cm2 = compute_acc_cm(y.argmax(dim=-1).cpu(), y_pred2.cpu())

        print(f"Accuracy: {acc}")
        print(f"Accuracy2:{acc2}")
        print(torch.allclose(cm, cm2))

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(cm.cpu(), cmap="Blues", annot=True, fmt="d")
        plt.show()
        plt.close()
