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


def get_data(batch_size=32, n_classes=10, n_rows=5, n_cols=5, train_size=1000, padding=0):
    """
    Returns
        x: tensor of shape (batch_size, 1, nrows * 32, n_cols * 32), dtype: float32
        y: tensor of shape (batch_size), dtype int64
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomAffine(0, (4/28,4/28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),

        ]),
        'test': transforms.Compose([
            transforms.Resize((32,32)),
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

    x_by_class = {d: x[y == d] for d in range(n_classes)}

    for _ in range(train_size):
        x_ = torch.zeros(batch_size, n_canvas, 32, 32)
        y_ = torch.zeros(batch_size, n_canvas, dtype=torch.int64)

        concentration_factor = Uniform(0.3,2).sample((batch_size,n_classes))
        concentration = torch.ones(n_classes)*concentration_factor

        probs_dir = Dirichlet(concentration).sample()
        digits = Multinomial(total_count=1, probs=probs_dir).sample((n_canvas,)).permute((1,0,2))

        for isample,sample in enumerate(digits):
            for id_,d in enumerate(sample):
                d = d.argmax().item()
                y_[isample,id_] = d

                probs_mult = torch.ones(len(x_by_class[d]))/len(x_by_class[d])

                x_[isample, id_] = x_by_class[d][Multinomial(total_count=1, probs=probs_mult).sample().argmax().item()]
        x_ = torch.nn.functional.fold(torch.transpose(x_.view(batch_size, n_canvas, 32 * 32), 2, 1), output_size=(32 * n_rows,
32 * n_cols), kernel_size=(32, 32), stride=(32, 32))
        yield x_, y_
