import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from tqdm import trange

from src.data.mnist import get_mnist_loaders

# I wrote this so the RealNVP can handle images better, because in the exercises it could barely handle 8x8 images,
# but MNIST is 28x28.
# I use a checkerboard split first (use every other pixel) for halving the tensors for the st-net, which itself is modified so that is not
# a simple MLP anymore but now a convolutional network which is better suited for images.
# After that, squeeze and and then it is possible to split easily by the channels.


# The squeeze layer is used to squeeze the image dimensions (28x28) into a smaller image but more channels.
class Squeeze(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x, y=None):
        B, C, H, W = x.shape
        f = self.factor
        x = x.reshape(B, C, H // f, f, W // f, f)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * f * f, H // f, W // f)
        return x, torch.zeros(B, device=x.device)  # logdet is 0 for reshaping

    def inverse(self, z, y=None):
        B, C, H, W = z.shape
        f = self.factor
        z = z.reshape(B, C // (f**2), f, f, H, W)
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.reshape(B, C // (f**2), H * f, W * f)
        return z


class ConvCondCouplingLayer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, condition_size, mask_type="checkerboard"
    ):
        super().__init__()
        self.mask_type = mask_type

        # for checkerboard, in_channels remains same but half are masked
        input_channels = (
            in_channels if mask_type == "checkerboard" else in_channels // 2
        )

        self.st_net = nn.Sequential(
            nn.Conv2d(input_channels + condition_size, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2 * input_channels, 3, padding=1),
        )

        # init with zeros so the st-net starts out as an identity transformation (for smoother learning)
        self.st_net[-1].weight.data.zero_()
        self.st_net[-1].bias.data.zero_()

    def get_mask(self, x):
        if self.mask_type == "checkerboard":
            B, C, H, W = x.shape
            mask = (
                torch.arange(H).view(-1, 1) + torch.arange(W)
            ) % 2  # create checkerboard with (H + W) % 2
            return mask.to(x.device).float().view(1, 1, H, W)
        else:
            # if the mask is not checkerboard then split the channels
            B, C, H, W = x.shape
            mask = torch.zeros(1, C, 1, 1, device=x.device)
            mask[:, : C // 2, :, :] = 1
            return mask

    def forward(self, x, y):
        mask = self.get_mask(x)
        x_masked = x * mask

        # Tile condition y (B, 10) to (B, 10, H, W) to concat with image
        y_geo = y.view(y.shape[0], y.shape[1], 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3]
        )

        # for channel split we only pass the active half to st_net
        st_input = (
            torch.cat([x_masked, y_geo], dim=1)
            if self.mask_type == "checkerboard"
            else torch.cat([x[:, : x.shape[1] // 2, :, :], y_geo], dim=1)
        )

        st = self.st_net(st_input)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # Numerical stability

        # apply mask to s and t so only half the input is transformed
        if self.mask_type == "checkerboard":
            inv_mask = 1 - mask
            z = x_masked + inv_mask * (x * torch.exp(s) + t)
            log_det = torch.sum(s * inv_mask, dim=[1, 2, 3])
        else:
            x1, x2 = x[:, : x.shape[1] // 2, :, :], x[:, x.shape[1] // 2 :, :, :]
            z2 = x2 * torch.exp(s) + t
            z = torch.cat([x1, z2], dim=1)
            log_det = torch.sum(s, dim=[1, 2, 3])

        return z, log_det

    def inverse(
        self, z, y
    ):  # again the same as in the forward pass but inverse, same principle as the simple RealNVP implementation above
        mask = self.get_mask(z)
        y_geo = y.view(y.shape[0], y.shape[1], 1, 1).expand(
            -1, -1, z.shape[2], z.shape[3]
        )

        if self.mask_type == "checkerboard":
            z_masked = z * mask
            st_input = torch.cat([z_masked, y_geo], dim=1)
            st = self.st_net(st_input)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            inv_mask = 1 - mask
            x = z_masked + inv_mask * ((z - t) * torch.exp(-s))
        else:
            z1, z2 = z[:, : z.shape[1] // 2, :, :], z[:, z.shape[1] // 2 :, :, :]
            st_input = torch.cat([z1, y_geo], dim=1)
            st = self.st_net(st_input)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            x2 = (z2 - t) * torch.exp(-s)
            x = torch.cat([z1, x2], dim=1)
        return x


# this class is for preprocessing the data
# because the images originally have integer values between 0 and 255, the distribution of them is sharp
# we want to add a bit of noies to each image just before passing it into our model just so the numbers have some kind of distribution around the integer values. this is called dequantization.
# then, we want to do a logit transform to be able to model the real numbers instead of just a specific range with hard cutoffs
class Preprocessing(nn.Module):
    def __init__(self, alpha=1e-5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y=None):
        noise = (
            torch.rand_like(x) / 255.0
        )  # the image was rescaled from [0, 255] to [0, 1] so we do the same transformation with the noise
        x = x + noise
        # to prevent the log from becoming -inf, we want to rescale the values of x to [0+alpha, 1-alpha]
        # for this we do:
        #       (((x - x.min) / x.max) * (1 - 2*alpha)) + alpha
        x = (((x - x.min()) / x.max()) * (1 - 2 * self.alpha)) + self.alpha
        z = torch.log(x) - torch.log(
            1 - x
        )  # logit(x) = log(x/(1-x)) = log(x) - log(1-x)

        # the derivative is 1 / (x * (1 - x))
        # log-det is then sum( log(1) - log(x) - log(1 - x) ) plus the factor
        log_det = -torch.sum(torch.log(x) + torch.log(1.0 - x), dim=[1, 2, 3])
        log_det += x[0].numel() * torch.log(
            torch.tensor((1 - 2 * self.alpha), device=x.device)
        )

        return z, log_det

    def inverse(self, z, y=None):
        # inverse logit is the sigmoid function
        x = torch.sigmoid(z)

        # inverse scaling (scale back to [0, 1])
        x = (x - x.min()) / x.max()  # neglect the alphas since they are very small

        return x


# activation normalization (instead of batch normalization) as in the glow paper (which is a kind of successor to the realnvp paper)
# unlike batch normalization, it learns the mean and variance during training
# it initializes the mean and variance during the first training batch
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            std = torch.std(x, dim=[0, 2, 3], keepdim=True)

            self.bias.data.copy_(-mean)
            self.weight.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)

    def forward(self, x, y=None):
        if self.initialized == 0:
            self.initialize(x)

        z = (x + self.bias) * self.weight

        # log-det is H * W * sum(log(weight))
        B, C, H, W = x.shape
        log_det = H * W * torch.sum(torch.log(torch.abs(self.weight) + 1e-6))

        return z, log_det.expand(B)  # expand log-det to fit batch size

    def inverse(self, z, y=None):
        return (z / self.weight) - self.bias


# invertible 1x1 convolutions replace the random rotation, which is responsible for mixing in all pixels into the transforms
class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init)  # initialize with orthogonal matrix

    def forward(self, x, y=None):
        B, C, H, W = x.shape
        log_det = H * W * torch.linalg.slogdet(self.weight)[1]
        z = F.conv2d(x, self.weight.view(C, C, 1, 1))
        return z, log_det.expand(B)

    def inverse(self, z, y=None):
        weight_inv = torch.inverse(self.weight)
        return F.conv2d(z, weight_inv.view(z.shape[1], z.shape[1], 1, 1))


# this is the final convolutional conditional realnvp implementation
class ConvCondRealNVPModule(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, condition_size: int = 10):
        super().__init__()
        self.condition_size = condition_size

        # first layer is preprocessing
        self.layers = nn.ModuleList(
            [
                Preprocessing(),
            ]
        )

        # first a few checkerboard- and then channel-splits for the rest
        rest = n_layers % 3
        for n in range((n_layers // 3) + rest):
            self.layers.append(ActNorm(1))
            self.layers.append(Invertible1x1Conv(1))
            self.layers.append(
                ConvCondCouplingLayer(1, hidden_size, condition_size, "checkerboard")
            )
        self.layers.append(Squeeze(factor=2))

        for n in range(n_layers // 3):
            self.layers.append(ActNorm(4))
            self.layers.append(Invertible1x1Conv(4))
            self.layers.append(
                ConvCondCouplingLayer(4, hidden_size, condition_size, "channel")
            )
        self.layers.append(Squeeze(factor=2))

        for n in range(n_layers // 3):
            self.layers.append(ActNorm(16))
            self.layers.append(Invertible1x1Conv(16))
            self.layers.append(
                ConvCondCouplingLayer(16, hidden_size, condition_size, "channel")
            )

        # register base dist parameters as buffers so they move the gpu, too
        self.register_buffer("loc", torch.zeros(16 * 7 * 7))
        self.register_buffer("cov", torch.eye(16 * 7 * 7))

    @property
    def base_dist(self):
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    @property
    def device(self):
        return self.loc.device

    def forward(self, x, y):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x, y.float())
            log_det_total += log_det

        z = x.reshape(x.shape[0], -1)
        return z, log_det_total

    def decode(self, z, y):
        z = z.reshape(z.shape[0], 16, 7, 7)
        for layer in reversed(self.layers):
            z = layer.inverse(z, y.float())
        return z

    def sample(self, n_samples: int = 1, batch_size: int = 64, device=None):
        if device is None:
            device = self.device

        # 1. Generate all the latent vectors (z) and labels (y) first
        z_list = [
            self.base_dist.sample((n_samples,)) for _ in range(self.condition_size)
        ]
        z_full = torch.vstack(z_list).to(device)

        y_full = (
            torch.nn.functional.one_hot(
                torch.arange(0, self.condition_size, device=device)
            )
            .repeat_interleave(n_samples, 0)
            .reshape((n_samples * self.condition_size, self.condition_size))
        ).float()
        decoded_samples = []
        # 2. Decode in chunks to prevent OOM
        with torch.no_grad():
            for i in trange(
                0,
                len(z_full),
                batch_size,
                desc="Generating new data... ",
                leave=True,
                disable=not sys.stdout.isatty(),
            ):
                z_batch = z_full[i : i + batch_size]
                y_batch = y_full[i : i + batch_size]

                # Decode just this chunk
                decoded_batch = self.decode(z_batch, y_batch)

                # Move the finished batch to CPU immediately so it doesn't crowd GPU memory
                decoded_samples.append(decoded_batch.cpu())
        # 3. Concatenate all the CPU tensors together
        return torch.cat(decoded_samples, dim=0).float().to(
            device=self.device
        ), y_full.to(device=self.device)

    def z_sample(self, n_samples: int = 100):
        return self.base_dist.sample((n_samples,))


class RealNVP(L.LightningModule):
    def __init__(self, model: ConvCondRealNVPModule = None, **kwargs):
        super().__init__()
        if model is None:
            model = ConvCondRealNVPModule(hidden_size=512, n_layers=12)
            model.to(device=kwargs.get("device", torch.device("cuda:0")))
            # model = torch.compile(model, fullgraph=True, mode='reduce-overhead')
        self.model = model

    def loss_fn(self, x, y):
        z, l = self.model(x, y)
        left = self.model.base_dist.log_prob(z)
        return -torch.mean(left + l)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(x, y)
        return loss

    def configure_optimizers(self, **kwargs):
        optimizer = optim.AdamW(self.parameters(), lr=kwargs.get("lr", 1e-3))
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(x, y)
        self.log("val_loss", loss)

    def sample(self, n: int = 1, **kwargs):
        return self.model.to(kwargs.get("device", torch.device("cuda"))).sample(
            n_samples=n,
            batch_size=kwargs.get("batch_size", 64),
            device=kwargs.get("device", torch.device("cuda:0")),
        )


def get_trained(**kwargs):
    L.pytorch.utilities.disable_possible_user_warnings()
    path = kwargs.get("path", Path(os.getcwd() + "/trained_models"))
    if kwargs.get("train", False):
        print("Training new classifier...")
        model = RealNVP()

        trainer = L.Trainer(
            logger=TensorBoardLogger(path, "realnvp_logs"),
            max_epochs=kwargs.get("max_epochs", 20),
            accelerator="cuda",
        )

        trl, tel = get_mnist_loaders(**kwargs)

        trainer.fit(model=model, train_dataloaders=trl, val_dataloaders=tel)
    else:
        model = RealNVP.load_from_checkpoint(
            path / Path("realnvp_logs/version_1/checkpoints/epoch=1-step=1876.ckpt")
        )  # hardcoded, good model

    print("Model loaded.")

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether to train a model or just load it from the latest checkpoint.",
    )
    parser.add_argument(
        "--max_epochs",
        action="store",
        type=int,
        default=10,
        help="How many epochs to train the CNN.",
    )
    parser.add_argument(
        "--fashion",
        action="store_true",
        default=False,
        help="Use the FashionMNIST dataset instead of the original MNIST.",
    )
    torch.set_float32_matmul_precision("high")
    args = parser.parse_args()
    args_dict = vars(args)

    m = get_trained(**args_dict)

    with torch.no_grad():
        s, y = m.sample(5000)
        print(s.shape, y.shape)

    from src.models.classification.cnn import get_trained as get_c, CNNClassifier

    c = get_c()
    t = L.Trainer()
    predictions = t.predict(
        c, dataloaders=DataLoader(TensorDataset(s, y), batch_size=128)
    )

    probslist, labelslist = zip(*predictions)
    y_pred = torch.cat(probslist)
    y_pred2 = torch.cat(labelslist)

    from src.metrics.prediction_accuracy import compute_metrics

    acc, cm = compute_metrics(y.argmax(dim=-1).cpu(), y_pred.cpu())
    acc2, cm2 = compute_metrics(y.argmax(dim=-1).cpu(), y_pred2.cpu())

    print(f"Accuracy: {acc}")
    print(f"Accuracy2:{acc2}")
    print(torch.allclose(cm, cm2))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(cm.cpu(), cmap="Blues", annot=True, fmt="d")
    plt.show()
    plt.close()
