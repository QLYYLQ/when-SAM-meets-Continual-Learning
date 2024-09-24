import numpy as np
from scipy.ndimage import label, measurements
from typing import Optional, Dict, List, Union, Tuple, Any
from torch import Tensor


def find_separated_boundaries(masks: Union[Tensor, np.array], label_list: Optional[List[List[int]]] = None,
                              min_size_percentage: int = 0.05) -> Tuple[List[Dict[int, Union[int, Any]]], List[Any]]:
    """
    找出每一块闭合的轮廓。

    参数:
    mask : Optional[Tensor,np.ndarray]
        shape:[B,1,H,W]
    label_list : Optional[List[List[int]]]
        list of labels in each mask, can be none
    min_size_percentage : int
        when mask size smaller than min_size_percentage% times total size, ignore it

    返回:
    Union[List[Dict[int,np.array]],List[np.array]]
        Dict
            batch中每个mask对应的label_mask和它们的细分遮罩
        List
            这些遮罩堆叠的结果
    """

    if type(masks) == Tensor:
        masks = masks.cpu().squeeze().numpy()

    if label_list is None:
        label_list = np.unique(masks)
    label_mask = []
    stacked_label_mask = []
    total_pixel = masks.shape[-2] * masks.shape[-1]
    min_size = total_pixel * min_size_percentage / 100.0
    for index in range(masks.shape[0]):
        mask = masks[index].copy()
        labels = label_list[index]
        transient_mask = {}
        stacked_mask = []
        for transient_label in labels:
            mask1 = mask.copy()
            mask1[mask != transient_label] = False
            mask1[mask == transient_label] = True
            labeled_mask, number = label(mask1)
            area_size = measurements.sum(mask1, labeled_mask, index=range(1, number + 1))
            for transient_index, size in enumerate(area_size):
                if size < min_size:
                    # 因为number是从1开始的，enumerate的迭代器是从0开始的
                    labeled_mask[labeled_mask == (transient_index + 1)] = False
            transient_mask[transient_label] = labeled_mask
            stacked_mask.append(labeled_mask)
        stacked_mask = np.stack(stacked_mask)
        label_mask.append(transient_mask)
        stacked_label_mask.append(stacked_mask)

    return label_mask, stacked_label_mask


def create_bbox_from_masks(mask):
    """[b,1,H,W]的label或者是[num_labels,1,H,W]的label拆分"""
    origin_label_bbox = []
    for origin_label in range(mask.shape[0]):
        origin_label_bbox.append(create_bbox(origin_label))

    raise NotImplementedError


def create_bbox(mask:np.array,value:Union[int,bool]):
    """传入的mask要是二维的"""
    true_y,true_x = np.where(mask == value)
    return np.array([[np.min(true_x),np.min(true_y),np.max(true_x),np.max(true_y)]])


if __name__ == '__main__':
    pass