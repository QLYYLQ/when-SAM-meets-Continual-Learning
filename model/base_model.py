import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, List, Tuple, Union


# 设计理念：在过程中各个组件除了必须以外不接受mask输入，所有和mask相关的操作统一在一个方法中实现

class BaseModel(nn.Module):
    def __init__(self, image_encoder: Optional[nn.Module] = None, text_encoder: Optional[nn.Module] = None,
                 prompt_encoder: Optional[nn.Module] = None, query_net: Optional[nn.Module] = None,
                 mask_decoder: Optional[nn.Module] = None):
        super(BaseModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.query_net = query_net
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, data: Dict[str, torch.Tensor], mask_prompt: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        image_temporary_features:Dict[int, torch.Tensor] = {}
        reshape_mask:List[torch.Tensor] = []

        images = data["image"]
        masks = data["mask"]
        text_prompt = data["text_prompt"]
        text_embeddings = self.text_encoder(text_prompt)
        image_temporary_features = self.image_encoder(images)
        mask_dict = self.align_mask_dimension(image_temporary_features,masks)

        image_embeddings = image_temporary_features[-1]
        mask_embeddings = self.prompt_encoder(mask_prompt)
        query_embeddings = self.query_net(image_embeddings, text_embeddings)
        mask = self.mask_decoder(image_embeddings=image_embeddings, image_pe=self.prompt_encoder.get_dense_pe(),
                                 sparse_prompt_embeddings=query_embeddings, dense_prompt_embeddings=mask_embeddings,
                                 multimask_output=False, hq_token_only=False,
                                 interm_embeddings=image_temporary_features)
        return mask

    def for_tensorboard(self):
        raise NotImplementedError

    @staticmethod
    def align_mask_dimension(image_dict:Dict[int,torch.Tensor], mask:torch.Tensor):
        """
        image_dict:Dict[stage:image]
        image. Shape = [B,C,H,W]; mask. Shape = [B,1,H,W]
        return:
            mask_dict={stage:masks};masks.shape=[B,1,H,W]
        """
        mask_dict:Dict[int,torch.Tensor]={}
        for stage,image in image_dict.items():
            mask_new = []
            transient_mask = F.interpolate(mask.float(),size = image.shape[-2:],mode="nearest").to(dtype=torch.bool)
            mask_new.append(transient_mask)
            mask_new = torch.cat(mask_new,dim=0)
            mask_dict[stage] = mask_new
        return mask_dict


class BaseSAM(nn.Module):
    def __init__(self, image_encoder: Optional[nn.Module] = None, prompt_encoder: Optional[nn.Module] = None,mask_Decoder: Optional[nn.Module] = None,**kwargs):
        super(BaseSAM, self).__init__(**kwargs)
        self.image_encoder = image_encoder
        self.Prompt_encoder = prompt_encoder
        self.Mask_Decoder = mask_Decoder