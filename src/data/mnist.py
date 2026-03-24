import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torchvision.datasets import MNIST, FashionMNIST

from pathlib import Path


def download_mnist(path: Path | None = None, **kwargs):
    """
    Downloads the MNIST dataset.

    Returns:
        train_mnist, test_mnist
    """
    if path is None:
        path = Path(os.getcwd())
        
    if kwargs.get('fashion', False):
        train_mnist = FashionMNIST(root=path, train=True, download=True)
        test_mnist = FashionMNIST(root=path, train=False, download=True)
    else:
        train_mnist = MNIST(root=path, train=True, download=True)
        test_mnist = MNIST(root=path, train=False, download=True)

    return train_mnist, test_mnist


def mnist_to_dataset(train_mnist: MNIST, test_mnist: MNIST, **kwargs):
    """
    Converts MNIST class to TensorDataset class.
    TensorDataset has the advantage that all of the data can be placed on the GPU which speeds up training significantly.
    """
    train_data = train_mnist.data
    train_targ = train_mnist.targets

    test_data = test_mnist.data
    test_targ = test_mnist.targets

    train_dataset = TensorDataset(train_data, train_targ)
    test_dataset = TensorDataset(test_data, test_targ)

    return train_dataset, test_dataset


def process_mnist(train_dataset: TensorDataset, test_dataset: TensorDataset, **kwargs):
    """
    Prepares MNIST data for training.
    Converts the data from (28, 28) to (1, 28, 28), to float, and puts them on the right device.
    Converts the targets to one-hot encoded and puts them on the same device.

    Returns 'updated' TensorDatasets.
    """
    device = kwargs.get(
        "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    training_data = train_dataset.tensors[0].to(device=device).unsqueeze(1).float()
    training_targ = F.one_hot(train_dataset.tensors[1].to(device=device).long()).float()

    testing_data = test_dataset.tensors[0].to(device=device).unsqueeze(1).float()
    testing_targ = F.one_hot(test_dataset.tensors[1].to(device=device).long()).float()

    if kwargs.get("rescale", True):  # rescales to [0, 1]
        training_data = training_data / training_data.max()
        testing_data = testing_data / testing_data.max()

    trainset = TensorDataset(training_data, training_targ)
    testset = TensorDataset(testing_data, testing_targ)

    return trainset, testset


def tds_to_dl(tds: TensorDataset, **kwargs):
    if tds.tensors[0].device == torch.device("cuda:0"):
        dl = DataLoader(
            tds,
            batch_size=kwargs.get("batch_size", 64),
            shuffle=kwargs.get("shuffle", True),
        )
    else:
        dl = DataLoader(
            tds,
            batch_size=kwargs.get("batch_size", 64),
            shuffle=kwargs.get("shuffle", True),
            num_workers=16,
            persistent_workers=True,
            pin_memory=True,
        )

    return dl


def processed_to_dataloaders(trs: TensorDataset, tes: TensorDataset, **kwargs):
    trl = tds_to_dl(trs, **kwargs)
    tel = tds_to_dl(
        tes, **kwargs, shuffle=False
    )  # overwrite shuffle to false for testing dataset

    return trl, tel


def get_mnist_sets(**kwargs):
    trm, tem = download_mnist(**kwargs)
    trs, tes = mnist_to_dataset(trm, tem, **kwargs)
    trs, tes = process_mnist(trs, tes, **kwargs)
    return trs, tes


def get_mnist_loaders(**kwargs):
    trs, tes = get_mnist_sets(**kwargs)
    trl, tel = processed_to_dataloaders(trs, tes, **kwargs)
    return trl, tel


if __name__ == "__main__":
    trl, tel = get_mnist_loaders()
