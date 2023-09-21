# -------CNN based predictor--------
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# Genome Barcode to genome matrix (picture of micrbial genome)
# ------construct CNN with residual learning structure----------
