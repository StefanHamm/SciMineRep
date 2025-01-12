import random
import torch
import numpy as np
import os


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# path to where this file is located
dir_path = os.path.dirname(os.path.realpath(__file__))