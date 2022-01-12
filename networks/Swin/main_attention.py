from .swin import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gen_swin():
    model = SwinTransformer()
    
    return model

class swin_main_attention(nn.Module):
  def __init__(self):
    super(swin_main_attention,self).__init__()
    self.swin = gen_swin()
    state_dict = torch.load('./weights/upernet_swin_tiny_patch4_window7_512x512.pth')['state_dict']
    new_dict = {}

    for key, value in state_dict.items():
      new_dict[key.replace('backbone.','')] = value

    msg = self.swin.load_state_dict(new_dict, strict=False)
    # print(msg)
    gamma = 2
    b = 1
    C = 672
    t = int(abs((math.log(C, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    self.attention_conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
    self.attention_avg_pool = nn.AdaptiveAvgPool2d(1)
    self.sigmoid = nn.Sigmoid()

    self.head = nn.Conv2d(672, 1, kernel_size=1, stride=1)
    self.activation = nn.ReLU()


  def forward(self, img):
      pre = self.swin(img)
      pre1 = pre[0]
      pre2 = pre[1]
      pre3 = pre[2]

      pre3 = F.interpolate(pre3, scale_factor=4)
      pre2 = F.interpolate(pre2, scale_factor=2)
      pre = torch.cat([pre1,pre2,pre3], dim=1)

      y = self.attention_avg_pool(pre)
      y = self.attention_conv(y.squeeze(-1).transpose(-1,-2))
      y = y.transpose(-1, -2).unsqueeze(-1)
      y = self.sigmoid(y)
      pre = pre * y.expand_as(pre)

      pre = self.head(pre)
      pre = F.interpolate(pre, scale_factor=4, mode='bicubic')
      pre = self.activation(pre)
      
      return pre