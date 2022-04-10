import matplotlib.pyplot as plt
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os


def plotWeights(
    weights, maxW, minW, figsize=(10, 6), title="receptive_field", display_fig=False
):
    plt.figure()
    plt.imshow(weights, vmin=minW, vmax=maxW, cmap="hot_r")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s.svg" % title)
    if display_fig:
        plt.show()


def plot_confusion_matrix(
    data, labels=np.arange(10), title="confusion_matrix", display_fig=False
):
    sns.set(color_codes=True)
    plt.figure()
    plt.title("Confusion Matrix")
    ax = sns.heatmap(
        data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Scale"}
    )
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(os.path.join("output", "%s.svg" % title))
    if display_fig:
        plt.show()


# Have a function for re-arranging weights and for plotting other weights


def ReshapeWeights(weights, n, assignments="none"):
    if n == 100:
        rows = 10
        cols = 10
        RWeights = torch.zeros(280, 280)
        for i in range(10):
            for j in range(28):
                RWeights[28 * i + j, :] = torch.reshape(
                    weights[10 * i : 10 * i + 10, 28 * j : 28 * j + 28], (1, 280)
                )
        if assignments != "none":
            assignments = torch.reshape(assignments, (rows, cols))
    elif n == 30:
        rows = 6
        cols = 5
        RWeights = torch.zeros(28 * rows, 28 * cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28 * i + j, :] = torch.reshape(
                    weights[cols * i : cols * (i + 1), 28 * j : 28 * (j + 1)],
                    (1, 28 * cols),
                )
        if assignments != "none":
            assignments = torch.reshape(assignments, (rows, cols))
    elif n == 50:
        rows = 10
        cols = 5
        RWeights = torch.zeros(28 * rows, 28 * cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28 * i + j, :] = torch.reshape(
                    weights[cols * i : cols * (i + 1), 28 * j : 28 * (j + 1)],
                    (1, 28 * cols),
                )
        if assignments != "none":
            assignments = torch.reshape(assignments, (rows, cols))
    elif n == 10:
        rows = 5
        cols = 2
        RWeights = torch.zeros(28 * rows, 28 * cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28 * i + j, :] = torch.reshape(
                    weights[cols * i : cols * (i + 1), 28 * j : 28 * (j + 1)],
                    (1, 28 * cols),
                )
        if assignments != "none":
            assignments = torch.reshape(assignments, (rows, cols))
    elif n == 300:
        rows = 15
        cols = 20
        RWeights = torch.zeros(28 * rows, 28 * cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28 * i + j, :] = torch.reshape(
                    weights[cols * i : cols * (i + 1), 28 * j : 28 * (j + 1)],
                    (1, 28 * cols),
                )

        if assignments != "none":
            assignments = torch.reshape(assignments, (rows, cols))

    return RWeights, assignments
