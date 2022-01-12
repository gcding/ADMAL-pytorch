from torch.nn.modules import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.autograd import Variable
from torchvision import models

class VGGNet(nn.Module):
  def __init__(self):
    super(VGGNet, self).__init__()
    self.select = ['26']
    self.vgg19 = models.vgg19(pretrained=True).features

  def forward(self, x):
    for name, layer in self.vgg19._modules.items():
      x = layer(x)
      if name in self.select:
        return x

class Counter(nn.Module):
  def __init__(self, model_name, mode = "DME"):
    super(Counter, self).__init__()
    self.model_name = model_name
    self.mode = mode

    if model_name == "ADML":
      if mode == "DMG":
        from .CNN import CNN as net
      else:
        from .CNN import CNN as net
        from .Swin import swin_main_attention as backbone
    elif model_name == "ASPDNet":
      from .ASPDNet import ASPDNet as backbone
    else:
      raise ValueError('Network {} cannot be recognized. Please define your own Network here.'.format(model_name))
    
    self.vgg19 = VGGNet()
    self.vgg19=self.vgg19.cuda()

    if mode == 'DMG':
      self.CCN = net()
      self.CCN = self.CCN.cuda()
    else:
      self.CCN = net()
      self.CCN = self.CCN.cuda()
      self.backbone = backbone()
      self.backbone = self.backbone.cuda()

  @property
  def loss(self):
    # The code will be added after the paper is accepted
    pass

  def build_loss(self):
    # The code will be added after the paper is accepted
    pass

  def forward(self, img, point_map):
    # The code will be added after the paper is accepted
    if self.mode == 'DMG':
      pass
    else:
      pass
  
  def test_forward(self, img, point_map):
    if self.mode == 'DMG':
      # The code will be added after the paper is accepted
      pass
    else:
      pred_map = self.backbone(img)
      return pred_map