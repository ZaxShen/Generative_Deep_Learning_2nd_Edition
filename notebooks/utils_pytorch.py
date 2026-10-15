import matplotlib.pyplot as plt
import numpy as np
import torch


def sample_batch(dataset):
    """
    Returns the images from the first batch of a PyTorch DataLoader.
    """
    batch = next(iter(dataset))

    if isinstance(batch, (tuple, list)):
        batch = batch[0]

    return batch.detach().cpu().numpy()


def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    """
    Displays n images from a PyTorch tensor or NumPy array.

    Supports:
        - CPU, MPS, and CUDA tensors
        - NCHW (PyTorch) and NHWC (TensorFlow) formats
        - Grayscale and RGB images
    """

    # Convert PyTorch tensor to NumPy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Convert NCHW → NHWC
    if images.ndim == 4 and images.shape[1] in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))

    # Normalize pixel values
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    n = min(n, len(images))

    plt.figure(figsize=size)

    for i in range(n):
        plt.subplot(1, n, i + 1)

        image = images[i].astype(as_type)

        # Remove channel dimension for grayscale
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        plt.imshow(image, cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()
