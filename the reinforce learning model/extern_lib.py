import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers, datasets,losses,optimizers,utils
import keras.preprocessing.sequence
import matplotlib.pyplot as plt
import sys
import math
import copy
import datetime
import random
import collections

#this file contains all the Hyperparameters

units=32
batch_size=32
max_input_length=10
gamma=0.9
learning_rate=0.001
update_coefficient=0.5