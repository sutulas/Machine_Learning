import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from lightning.pytorch.loggers import CSVLogger
from matplotlib.pyplot import subplots
from pytorch_lightning import Trainer, seed_everything
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.multiprocessing import set_sharing_strategy
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from torchinfo import summary
from torchmetrics import MeanAbsoluteError, R2Score

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
set_sharing_strategy("file_system")

from ISLP.torch import (
    ErrorTracker,
    SimpleDataModule,
    SimpleModule,
    rec_num_workers,
)
from ISLP.torch.imdb import (
    load_lookup,
    load_sequential,
    load_sparse,
    load_tensor,
)
from torchvision.datasets import CIFAR100, MNIST
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor

max_num_workers = rec_num_workers()


def summary_plot(
    results,
    ax,
    col="loss",
    valid_legend="Validation",
    training_legend="Training",
    ylabel="Loss",
    fontsize=20,
):
    for column, color, label in zip(
        [f"train_{col}_epoch", f"valid_{col}"],
        ["black", "red"],
        [training_legend, valid_legend],
    ):
        results.plot(
            x="epoch", y=column, label=label, marker="o", color=color, ax=ax
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    return ax
