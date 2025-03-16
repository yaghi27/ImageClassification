import sys
import os
src_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
sys.path.append(src_folder)


from torchvision.models import resnet50
from torch import nn
from DataLoader import DataLoader
from ModelTrainer import ModelTrainer
from Utils.utilities import *
from Gui import Gui