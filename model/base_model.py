import torch
from torch import nn
from typing import Optional


class BaseModel(nn.Module):
    def __init__(self, image_encoder: Optional[nn.Module] = None, text_encoder: Optional[nn.Module] = None,
                 query_net: Optional[nn.Module] = None, mask_decoder: Optional[nn.Module] = None):
        super(BaseModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.query_net = query_net
        self.mask_decoder = mask_decoder

    def forward(self, images: torch.Tensor, text_prompt: str):
        text_embeddings = self.text_encoder(text_prompt)
        image_embeddings = self.image_encoder(images)
        query_embeddings = self.query_net(image_embeddings, text_embeddings)
        mask = self.mask_decoder(query_embeddings, image_embeddings)
        return mask
