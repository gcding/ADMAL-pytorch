import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d
from torchvision import models
from misc.utils import *

class CNN(nn.Module):
  def __init__(self, pretrained=True):
    super(CNN, self).__init__()
    # The code will be added after the paper is accepted
    pass
    initialize_weights(self.modules())

  def forward(self, feature ,x):
    # The code will be added after the paper is accepted
    pass

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, std=0.01)
        if m.bias is not None:
          m.bias.data.fill_(0)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.fill_(1)
        m.bias.data.fill_(0)
