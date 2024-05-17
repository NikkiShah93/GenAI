## first the imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path 

## general params
PATH_DIR = Path('../data/')
PATH_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = 'wizardOfOz.txt'
FILE_PATH = PATH_DIR / FILE_NAME

## the main hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-3
evaluation_interval = 300
epochs = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## then the same script that was developed in the notebook

## reading the file
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()