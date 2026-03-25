import torchmetrics
import torch


def compute_fid():
    pass


# more complicated than i wanted it to be, because FID works with an Inception V3 model which takes 299x299 images, but MNIST is 28x28.
# I'm not sure if that's a good metric for MNIST
