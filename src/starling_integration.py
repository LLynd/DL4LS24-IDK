import numpy as np
import anndata as ad
import pandas as pd
import torch
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping  # ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from starling import starling, utility
from src.dataloader import CustomDataLoader


def get_starling_model_and_data(config):
    seed_everything(config.seed, workers=True)
    cdl = CustomDataLoader(config)
    
    if config.test_size is not None:
        X_train, Y_train, X_test, Y_test = cdl.get_data(preprocess=True)
    elif config.test_size is None:
        X_train, Y_train = cdl.get_data(preprocess=True)
    
    adata = utility.init_clustering("KM", data, k=20)
    st = starling.ST(adata)

def classify_starling_clusters():
    pass

def train_starling():
    pass
