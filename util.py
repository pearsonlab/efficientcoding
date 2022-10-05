import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("agg")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def kernel_images(W, kernel_size, image_channels, rows=None, cols=None, spacing=1):
    """
    Return the kernels as tiled images for visualization
    :return: np.ndarray, shape = [rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing, 1]
    """

    W /= np.linalg.norm(W, axis=0, keepdims=True)
    W = W.reshape(image_channels, -1, W.shape[-1])

    if rows is None:
        rows = int(np.ceil(math.sqrt(W.shape[-1])))
    if cols is None:
        cols = int(np.ceil(W.shape[-1] / rows))

    kernels = np.ones([3, rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x:x+kernel_size, y:y+kernel_size] = kernel

    return kernels.clip(0, 1)


def plot_convolution(weight: torch.Tensor):
    if torch.is_tensor(weight):
        weight = weight.numpy()
    weight = weight / np.linalg.norm(weight, axis=-1, keepdims=True)

    fig = plt.figure(figsize=(4, 4))
    plt.plot(weight[:, 0, :].T)
    plt.tight_layout()
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncol, nrow = fig.canvas.get_width_height()
    buf = buf.reshape(ncol, nrow, 3)
    plt.close()

    return buf.transpose(2, 0, 1)

