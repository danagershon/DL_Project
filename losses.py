
import torch


def reconstruction_loss_MSE(x, x_rec):
    """
    Compute the mean squared error (MSE) between original and reconstructed images.

    :param x: Original images
    :param x_rec: Reconstructed images
    :return: MSE loss
    """
    return torch.mean((x - x_rec) ** 2)


def reconstruction_loss_BCE(x, x_rec):
    """
    Compute the binary cross entropy (BCE) between original and reconstructed images.

    :param x: Original images
    :param x_rec: Reconstructed images
    :return: BCE loss
    """
    # Add a channel dimension to x so it matches x_rec's shape
    x = x.unsqueeze(1)  # Shape will become [batch_size, 1, 28, 28]
    return torch.nn.functional.binary_cross_entropy(x_rec, x)
