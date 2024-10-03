import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AllocationDataset(Dataset):
    