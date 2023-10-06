import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import sys, os
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
try:
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to Numpy')
import matplotlib.pyplot as plt

from transformer.modules import Encoder, Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD, Noam
from transformer.losses import CrossEntropy
from transformer.prepare_data import DataPreparator



