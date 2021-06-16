import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plotWeights(weights, maxW, minW, figsize=(10, 6), title='receptive_field', display_fig=False):

    # _, axes = plt.subplots(1, 2, figsize=figsize)
    # color = plt.get_cmap('RdBu', 11)
    print(weights.shape)
    plt.imshow(weights, vmin=minW, vmax=maxW)
    # , axes[1].matshow(
    #     assignments, cmap=color, vmin=-1.5, vmax=9.5)
    # divs = make_axes_locatable(axes[0]), make_axes_locatable(axes[1])
    # caxs = divs[0].append_axes("right", size="5%", pad=0.05), divs[1].append_axes(
    #     "right", size="5%", pad=0.05)

    # plt.colorbar()
    # plt.colorbar(ims[1], cax=caxs[1], ticks=np.arange(-1, 10))
    plt.tight_layout()
    plt.savefig('%s.svg' % title)
    if display_fig:
        plt.show()


def plotConfusion(weights, confusion, maxW, minW, figsize=(10, 6)):
    _, axes = plt.subplots(1, 2, figsize=figsize)

    color = plt.get_cmap('RdBu', 11)
    ims = axes[0].imshow(weights, cmap='hot_r', vmin=minW, vmax=maxW), axes[1].matshow(
        confusion, cmap='hot_r', vmin=-0, vmax=torch.max(confusion))
    divs = make_axes_locatable(axes[0]), make_axes_locatable(axes[1])
    caxs = divs[0].append_axes("right", size="5%", pad=0.05), divs[1].append_axes(
        "right", size="5%", pad=0.05)

    plt.colorbar(ims[0], cax=caxs[0])
    plt.colorbar(ims[1], cax=caxs[1], ticks=np.arange(-1, 10))
    plt.tight_layout()
    plt.show()

# Have a function for re-arranging weights and for plotting other weights


def ReshapeWeights(weights, n, assignments='none'):
    if (n == 100):
        rows = 10
        cols = 10
        RWeights = torch.zeros(280, 280)
        for i in range(10):
            for j in range(28):
                RWeights[28*i+j, :] = torch.reshape(
                    weights[10*i:10*i+10, 28*j:28*j+28], (1, 280))
        if assignments != 'none':
            assignments = torch.reshape(assignments, (rows, cols))
    elif (n == 30):
        rows = 6
        cols = 5
        RWeights = torch.zeros(28*rows, 28*cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28*i+j, :] = torch.reshape(
                    weights[cols*i:cols*(i+1), 28*j:28*(j+1)], (1, 28*cols))
        if assignments != 'none':
            assignments = torch.reshape(assignments, (rows, cols))
    elif (n == 50):
        rows = 10
        cols = 5
        RWeights = torch.zeros(28*rows, 28*cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28*i+j, :] = torch.reshape(
                    weights[cols*i:cols*(i+1), 28*j:28*(j+1)], (1, 28*cols))
        if assignments != 'none':
            assignments = torch.reshape(assignments, (rows, cols))
    elif (n == 10):
        rows = 5
        cols = 2
        RWeights = torch.zeros(28*rows, 28*cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28*i+j, :] = torch.reshape(
                    weights[cols*i:cols*(i+1), 28*j:28*(j+1)], (1, 28*cols))
        if assignments != 'none':
            assignments = torch.reshape(assignments, (rows, cols))
    elif (n == 300):
        rows = 15
        cols = 20
        RWeights = torch.zeros(28*rows, 28*cols)
        for i in range(rows):
            for j in range(28):
                RWeights[28*i+j, :] = torch.reshape(
                    weights[cols*i:cols*(i+1), 28*j:28*(j+1)], (1, 28*cols))

        if assignments != 'none':
            assignments = torch.reshape(assignments, (rows, cols))

    return RWeights, assignments
