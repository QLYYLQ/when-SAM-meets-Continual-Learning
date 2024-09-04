import numpy as np
import os.path as osp

def filter_images(dataset, labels, labels_old, overlap):
    """
    Filter images according to the overlap.
    create a list of images and mask paths which is use for filtering.
    Also used to create the index_list.txt from tasks in config

    Parameters
    ----------
    dataset
    labels
    labels_old
    overlap

    Returns
    -------

    """
    index = []
    if 0 in labels:
        labels.remove(0)
    labels_cum = [x for x in labels + labels_old if x != 0] + [0]
    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        # 这里说的是：只要图片中包含有新类别就好
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i]["data"][1]))
        if fil(cls):
            index.append((dataset[i]["path"][0],dataset[i]["path"][1]))
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return index

def save_list_from_filter(index,save_path):
    with open(save_path,"w") as f:
        for pair in index:
            f.write(f"{pair[0]},{pair[1]}\n")

def load_list_from_path(index,save_path):
    new_list = []
    with open(save_path,"r") as f:
        for line in f:
            x = line[:-1].split(",")
            new_list.append((x[0],x[1]))