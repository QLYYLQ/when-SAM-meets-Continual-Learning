import numpy as np


def filter_images(dataset, labels, labels_old, overlap):
    """
    Filter images according to the overlap.
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
            index.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return index

