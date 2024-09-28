import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from typing import List, Union, Dict
from dataset import Dataloader
from torch.utils.data import DataLoader as PytorchDataloader
from munch import Munch
from utils.finetune_sam.create_finetune_sam_point_and_bbox import (find_separated_boundaries,
                                                                   create_bbox_from_origin_and_separated_label,
                                                                   create_point_from_origin_and_separated_label,
                                                                   create_point_label_from_origin_and_separated_points
                                                                   )
from tqdm import tqdm


def build_optimizer(model: nn.Module, optim_cfg: Union[Dict, Munch]):
    assert "type" in optim_cfg
    _optim_config = optim_cfg.copy()
    optim_type = _optim_config.pop("type")
    optim = getattr(torch.optim, optim_type)(filter(lambda p: p.requires_grad, model.parameters()), **_optim_config)
    return optim


def train_function(model: Dict[str, nn.Module], train_loader: Union[Dataloader, PytorchDataloader],
                   val_loader:Union[Dataloader,PytorchDataloader],optimizer_config: Union[Dict, Munch],
                   epoch: int, writer: SummaryWriter):
    if model["finetune_part"] is None:
        raise ValueError("model['finetune_part'] is not defined, please define a mask decoder to finetune")
    if model["image_encoder"] is None:
        raise ValueError("model['image_encoder'] is not defined, please define a image encoder")
    if model["prompt_encoder"] is None:
        raise ValueError("model['prompt_encoder'] is not defined, please define a prompt encoder")
    model = model["finetune_part"]
    image_encoder = model["image_encoder"]
    prompt_encoder = model["prompt_encoder"]
    image_encoder = image_encoder.eval()
    prompt_encoder = prompt_encoder.eval()
    model = model.train()
    if torch.cuda.is_available():
        image_encoder = image_encoder.cuda()
        prompt_encoder = prompt_encoder.cuda()
        model = model.cuda()
    optim = build_optimizer(model, optimizer_config)
    stage = 1
    for i in range(epoch):
        for batch in train_loader:
            image = batch["image"]
            label = batch["label"]
            mask = batch["mask"]
            label_index = batch["label_index"]
            image_embeddings, interim_embeddings = image_encoder(image)
            separated_label_dict,separated_label_list = find_separated_boundaries(label)
            origin_bbox, separated_bbox = create_bbox_from_origin_and_separated_label(label,separated_label_list)
            origin_foreground_points,origin_background_points,separated_foreground_points,separated_background_points =create_point_from_origin_and_separated_label(label,separated_label_list,"number",foreground_number=5,background_number=5)
            points_label11,points_label12,points_label_21,points_label22 = create_point_label_from_origin_and_separated_points(origin_foreground_points,separated_foreground_points)



            if stage == 1:
                pass

            elif stage == 2:
                pass

    raise NotImplementedError


def val_function():
    raise NotImplementedError

if __name__ == '__main__':
    pass