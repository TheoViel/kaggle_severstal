import numpy as np
import torch

seed = 2019

TRAIN_IMG_PATH = '../input/train_images/'
TEST_IMG_PATH = '../input/test_images/'
DATA_PATH = '../input/'
CP_PATH = '../checkpoints/'

IMG_SHAPE = (256, 1600)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4

VAL_BS = 1  # seresnext
# VAL_BS = 4  #resnet

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")