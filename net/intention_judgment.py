import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# DEVICE = torch.device('cpu')

class IntentionNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)