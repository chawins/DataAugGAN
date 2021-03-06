import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import keras
from keras.optimizers import SGD, RMSprop
from keras.utils.generic_utils import Progbar
from PIL import Image

# Define global parameters, treated like config file

VIS_DIR = "./vis/"
WEIGHT_DIR = "./weights/"

# MNIST params
SCALE = 127.5               # Data scaling factor
LATENT_SIZE = 100           # Dimensions of latent variables
INPUT_SHAPE = (28, 28, 1)   # Shape of input data
N_CLASSES = 10              # Number of classes/labels

N_EPOCH = 50                # Max. epochs for training
BATCH_SIZE = 128            # Batch size
# Number of iter of discriminator training per one iter of generator training
N_DIS = 5

# CIFAR-10 params
# SCALE = 127.5               # Data scaling factor
# LATENT_SIZE = 110           # Dimensions of latent variables
# INPUT_SHAPE = (32, 32, 3)   # Shape of input data
# N_CLASSES = 10              # Number of classes/labels

# N_EPOCH = 100               # Max. epochs for training
# BATCH_SIZE = 100            # Batch size
# # Number of iter of discriminator training per one iter of generator training
# N_DIS = 3
