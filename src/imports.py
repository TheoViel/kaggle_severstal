import fastai
from fastai.vision import *

import re
import gc
import os
import cv2
import sys
import time
import math
import pickle
import random
import operator
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import albumentations as albu
import matplotlib.pyplot as plt

from math import ceil
from PIL import Image
from datetime import date
from sklearn.metrics import *
from collections import Counter
from itertools import  filterfalse
from sklearn.model_selection import *
#from tqdm import tqdm_notebook as tqdm
tqdm = progress_bar
# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from torch import Tensor
from torch.nn.modules.loss import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import * 
from torchvision.models.resnet import *
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import BasicBlock

from pretrainedmodels.models.torchvision_models import pretrained_settings 
from pretrainedmodels.models.senet import pretrained_settings as pretrained_settings_senet
from pretrainedmodels.models.senet import SENet, SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck


