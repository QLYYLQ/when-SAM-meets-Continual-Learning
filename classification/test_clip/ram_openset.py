'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
'''
import argparse
import numpy as np

import torch

from PIL import Image
from CSS_Filter.ram.models import ram_plus
from CSS_Filter.ram import inference_ram_openset as inference
from CSS_Filter.ram import get_transform

from CSS_Filter.ram.utils import build_openset_llm_label_embedding
from torch import nn
import json

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default=r'D:\project\CSS_Filter\css_dataset\data\PascalVOC12\JPEGImages\2010_004072.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default=r'')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')
parser.add_argument('--llm_tag_des',
                    metavar='DIR',
                    help='path to LLM tag descriptions',
                    default=r'D:\project\ram\recognize-anything\datasets\tag_descriptions.json')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=r"D:\project\CSS_Filter\ram\pretrained\ram_plus_swin_large_14m.pth",
                     image_size=384,
                     vit='swin_l')

    #######set openset interference

    print('Building tag embedding:')
    with open(args.llm_tag_des, 'rb') as fo:
        llm_tag_des = json.load(fo)
    # print(llm_tag_des)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    model.tag_list = np.array(openset_categories)

    model.label_embed = nn.Parameter(openset_label_embedding.float())

    model.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    model.class_threshold = torch.ones(model.num_class) * 0.55
    #######

    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)
    print("Image Tags: ", res)
