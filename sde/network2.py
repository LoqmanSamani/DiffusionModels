import torch
import torch.nn as nn



"""neural network for the VE SDE (NCSN++)"""

class NCSNpp(nn.Module):
    def __init__(self, config):
        super().__init__()


    def forward(self, x):
        return x