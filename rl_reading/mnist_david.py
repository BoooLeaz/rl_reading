from tqdm import tqdm
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multinomial import Multinomial
from torch.distributions.uniform import Uniform

from copy import deepcopy

import torchvision
from torchvision import transforms, models

import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils

import torch.optim as optim


def get_loss_f():
    return nn.MSELoss()


class Grid:

    def __init__(self, n_rows, n_cols, padding, cell_height, cell_width):
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.padding = padding
            self.n_canvas = n_rows*n_cols
            self.cell_height = cell_height
            self.cell_width = cell_width
            self.shape = (
                (self.n_rows * self.cell_height) + ((self.n_rows+1) * self.padding),
                (self.n_cols * self.cell_width) + ((self.n_cols+1) * self.padding)
            )

    def bb(self, ind):
        row = ind//self.n_cols
        col = (ind%self.n_cols)//self.n_rows

        top = (row * self.cell_height) + ((row+1) * self.padding)
        left = (col * self.cell_width) + ((col+1) * self.padding)
        bottom = top + self.cell_height
        right = left + self.cell_width

        return top, left, bottom, right


def get_data(batch_size=32, n_classes=10, n_rows=5, n_cols=5, train_size=1000, padding=0):
    """
    Returns
        x: tensor of shape (batch_size, 1, nrows * 28, n_cols * 28), dtype: float32
        y: tensor of shape (batch_size), dtype int64
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((30,30)),
            transforms.RandomAffine(0, (4/28,4/28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),

        ]),
        'test': transforms.Compose([
            transforms.Resize((30,30)),
            transforms.RandomAffine(0, (4/28,4/28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
    }

    image_datasets = {
        "train" : torchvision.datasets.MNIST(
            'dataset/',
            train=True,
            download=True,
            transform=data_transforms["train"]
        ),
        "test" : torchvision.datasets.MNIST(
            'dataset/',
            train=False,
            download=True,
            transform=data_transforms["test"]
        )
    }
    
    x, y = zip(*[image_datasets["train"][i] for i in range(len(image_datasets["train"]))])
    x = torch.cat(x)
    y = torch.tensor(y).to(torch.int64)
    n_canvas = n_rows*n_cols

    x_by_class = {
        d:x[y==d] for d in range(n_classes)
    }

    grid = Grid(n_rows, n_cols, padding, *x[0].shape)

    for _ in range(train_size):

        x_ = torch.zeros(batch_size, *grid.shape)-1
        y_ = torch.zeros(batch_size, n_canvas, dtype=torch.int64)

        masks = torch.zeros(batch_size, n_classes, *grid.shape)-1

        concentration_factor = Uniform(0.3,2).sample((batch_size,n_classes))
        concentration = torch.ones(n_classes)*concentration_factor

        probs_dir = Dirichlet(concentration).sample()
        digits = Multinomial(total_count=1, probs=probs_dir).sample((n_canvas,)).permute((1,0,2))

        for isample,sample in enumerate(digits):
            for id_,d in enumerate(sample):
                d = d.argmax().item()
                y_[isample,id_] = d

                probs_mult = torch.ones(len(x_by_class[d]))/len(x_by_class[d])
                t,l,b,r = grid.bb(id_)

                masks[isample, d].narrow(0,t, grid.cell_height).narrow(1,l,grid.cell_width).copy_(x_by_class[d][
                    Multinomial(total_count=1, probs=probs_mult).sample().argmax().item()
                ])
                x_[isample].narrow(0,t, grid.cell_height).narrow(1,l,grid.cell_width).copy_(
                    masks[isample, d].narrow(0,t, grid.cell_height).narrow(1,l,grid.cell_width)
                )

        x_ = x_.view(-1, 1, *x_[0].shape)
        yield x_, y_
