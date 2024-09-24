import torch
from torch import nn

class BaseBackbone(nn.Module):
    def __init__(self,image_encoder,position_embedding):
        super(BaseBackbone,self).__init__()
        self.image_encoder = image_encoder