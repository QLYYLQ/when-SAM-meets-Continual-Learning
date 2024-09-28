import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Union
from functools import wraps
from torch import Tensor


def check_input(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(image, masks, *args, **kwargs):
        if isinstance(image, Tensor):
            image = image.cpu().numpy()
            image = np.transpose(image, axes=[1, 2, 0])
        if isinstance(masks, Tensor):
            masks = masks.cpu().numpy()
        if kwargs.get("type",None) is None:
            raise ValueError("must specify type")
        else:
            if kwargs["type"] not in ["origin", "separated"]:
                raise ValueError("type must be either 'origin' or 'separated'")
        if (kwargs.get("points", None) is not None) and (kwargs.get("bboxs", None) is not None):
            # if is origin label's bbox
            if isinstance(kwargs["bboxs"], np.ndarray):
                if kwargs["bboxs"].shape[0] != kwargs["points"].shape[0]:
                    raise ValueError("Both points and bboxs must have the same shape")
            if isinstance(kwargs["bboxs"], list):
                if len(kwargs["bboxs"]) != kwargs["points"].shape[0]:
                    raise ValueError("Both points and bboxs must have the same shape")
        return fn(image, masks, *args, **kwargs)

    return wrapper


def check_color(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(**kwargs):
        if kwargs.get("color_info") is not None:
            if len(kwargs["color_info"]) != 4:
                raise ValueError("color_info must have 4 elements")
        return fn(**kwargs)

    return wrapper


def create_color_list(color_number) -> np.ndarray:
    color = []
    for i in range(color_number):
        color.append(np.concatenate([np.random.random(3), np.array([0.6])]))
    return np.stack(color)


def show_mask(mask, ax, random_color=False,color_info=None):
    """mask is a true false mask with shape:[H,W]"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    if color_info is not None:
        if len(color_info) != 4:
            raise ValueError("color_info must have 4 elements, with rgb and transparency in [0,1]")
    h, w = mask.shape[-2:]
    if color_info is None:
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    else:
        mask_image = mask.reshape(h, w, 1) * color_info.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, label, ax,color_info, marker_size=100):
    edge_color = ["red","green"]
    color = color_info[:-1]
    ax.scatter(coords[:, 0], coords[:, 1], color=color, marker='*', s=marker_size, edgecolor=edge_color[label],
               linewidth=1.25)



def show_box(box, ax, color_info, number=None):
    if color_info is None:
        color_info = [0,1.0,0,1]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    edge_color = color_info[:3]
    alpha = color_info[3]
    ax.add_patch(plt.Rectangle((x0+4, y0+4), w-8, h-8, edgecolor=edge_color, facecolor=(0,0,0,0), lw=2))
    if number is not None:
        ax.text(x0+8, y0+20, str(number), color=edge_color, fontweight='bold', fontsize='small')


@check_input
def show_image_with_mask_and_prompt(image,masks, file_name, foreground_points=None,background_points=None, bboxs=None, mask_value = [0],type=None):
    """
    masks has the shape with the shape: [1,h,w] or [labels_number,1,h,w]
    points with the shape: [batch_size,points_number,2]
    bbox with the shape:[batch_size,bbox_number,1,4]
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if type == "origin":
        for mask in masks:
            mask = mask.squeeze()
            value_list = np.unique(mask)
            value_list = [x for x in value_list if x not in mask_value]
            color_list = create_color_list(len(value_list))
            for index,value in enumerate(value_list):
                mask1 = mask.copy()
                mask1[mask == value] = True
                mask1[mask != value] = False
                show_mask(mask1, plt.gca(), False,color_list[index])
                if bboxs is not None:
                    bbox = bboxs[index]
                    show_box(bbox, plt.gca(), color_list[index],index)
                if foreground_points is not None:
                    foreground_point = foreground_points[index]
                    background_point = background_points[index]
                    show_points(foreground_point,1,plt.gca(),color_list[index])
                    show_points(background_point,0,plt.gca(),color_list[index])
    elif type == "separated":
        color_list = create_color_list(masks.shape[0])
        for index,mask in enumerate(masks):
            mask = mask.squeeze()
            value_list = np.unique(mask)
            value_list = [x for x in value_list if x not in mask_value]
            for value in value_list:
                mask1 = mask.copy()
                mask1[mask == value] = True
                mask1[mask != value] = False
                show_mask(mask1, plt.gca(), False, color_list[index])
                bbox = bboxs[index]
                if bbox is not None:
                    for bbox1 in bbox:
                        show_box(bbox1, plt.gca(), color_list[index],index)
                foreground_point = foreground_points[index]
                background_point = background_points[index]
                if foreground_point is not None:
                    for point in foreground_point:
                        show_points(point,1,plt.gca(),color_list[index])
                if background_point is not None:
                    for point in background_point:
                        show_points(point,0,plt.gca(),color_list[index])
    plt.axis('off')
    plt.savefig(file_name + '_' + "origin" + '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
