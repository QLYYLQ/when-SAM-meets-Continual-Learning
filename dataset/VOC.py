# 每次迭代需要返回一个字典：图片，遮罩，任务编号，label的编号，text prompt的内容


import copy
import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision as tv
from PIL import Image
from torch import distributed
from .utils.filter_images import filter_images
from .register import register_training_dataset,register_evaluating_dataset

# classes = {
#     0: 'background',
#     1: 'aeroplane',
#     2: 'bicycle',
#     3: 'bird',
#     4: 'boat',
#     5: 'bottle',
#     6: 'bus',
#     7: 'car',
#     8: 'cat',
#     9: 'chair',
#     10: 'cow',
#     11: 'dining table',
#     12: 'dog',
#     13: 'horse',
#     14: 'motorbike',
#     15: 'person',
#     16: 'potted plant',
#     17: 'sheep',
#     18: 'sofa',
#     19: 'train',
#     20: 'tv monitor'
# }


@register_training_dataset
class Segmentation(data.Dataset):
    """
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, is_aug=True,transform=None):
        """
        删除：image_set='train'
        因为测试使用别的数据集（调控是否只返回特定的图片）
        Parameters
        ----------
        root
        is_aug
        transform
        """
        self.root = os.path.expanduser(root)
        self._check_dataset_exists()
        self.transform = transform
        # self.image_set = image_set
        splits_dir = os.path.join(self.root, 'splits')
        if is_aug: # and image_set == 'train':
            mask_dir = os.path.join(self.root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, "train" + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" '
                f'{split_f}'
            )

        self.images = self.load_data_path_to_list(split_f)

    def _check_dataset_exists(self):
        # 检测是否存在数据集
        if not os.path.isdir(self.root):
            raise RuntimeError(
                'Dataset not found or corrupted.'
            )

    def load_data_path_to_list(self, path):
        images = []
        with open(path, "r") as f:
            for line in f:
                x = line[:-1].split(' ')
                images.append((
                    os.path.join(self.root, "VOCdevkit/VOC2012", x[0][1:]),
                    os.path.join(self.root, x[1][1:])
                ))

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        target = Image.open(self.images[index][1])
        img = Image.open(self.images[index][0]).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)
        return {"data":[img, target],"path":[self.images[index][0],self.images[index][1]]}

    def viz_getter(self, index):
        image_path = self.images[index][0]
        raw_image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(raw_image, target)
        else:
            img = copy.deepcopy(raw_image)
        return image_path, raw_image, img, target

    def __len__(self):
        return len(self.images)


@register_training_dataset
class Increment(data.Dataset):

    def __init__(
            self,
            root,
            train=True,
            transform=None,
            order = None,
            labels=None,
            labels_old=None,
            idxs_path=None,
            masking=True,
            overlap=True,
            data_masking="current",
            **kwargs
    ):

        full_voc = Segmentation(root, 'train' if train else 'val', is_aug=True, transform=None)
        self.order = order
        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []
            # 去除labels中的0号
            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            # 下面这一坨注释是：确保了在学习新任务时，模型能够保留足够的旧任务示例，从而在持续学习过程中保持对旧任务的记忆。它通过平衡每个类别的
            # 示例数量来防止灾难性遗忘，同时也考虑了数据集之间可能存在的重叠，如果用sam的话我们应该不需要这个流程


            # if ssul_exemplar_path is not None:
            #     print(ssul_exemplar_path)
            #     if os.path.exists(ssul_exemplar_path):
            #         ssul_exemplar_idx_cls = torch.load(ssul_exemplar_path)
            #         for k in ssul_exemplar_idx_cls:
            #             if opts.method != 'SATS':
            #                 idxs += ssul_exemplar_idx_cls[k][:7]
            #             else:
            #                 idxs += ssul_exemplar_idx_cls[k]
            #         print('length of ssul-m balanced exemplar samples:', len(idxs))
            #     if len(idxs) == 0 or not os.path.exists(ssul_exemplar_path) and False:
            #         print(f'current task:{labels} ssul building exemplar set!')
            #         per_task_exemplar = opts.ssul_m_exemplar_total / len(labels_old)
            #         print(f'every class {per_task_exemplar} samples')
            #         assert idxs_path is not None
            #         idxs_old = np.load(idxs_path).tolist()
            #
            #         old_exemplar = (f'./balance_step_exemplar/{opts.dataset}_{opts.task}_step_{opts.step - 1}_exemplar'
            #                         f'.pth')
            #         ssul_exemplar_idx_cls = torch.load(old_exemplar)
            #         lens = {}
            #         for label in labels_old:
            #             ssul_exemplar_idx_cls[label - 1] = []
            #             lens[label - 1] = 0
            #         for idx in tqdm(idxs_old):
            #             img_cls = np.unique(np.array(full_voc[idx][1]))
            #             fg = 1
            #             print(lens)
            #             for label in img_cls:  #QUESTION??? NOT EVERY CLASS TO BE 20 SAMPLES
            #                 if label in ssul_exemplar_idx_cls.keys() and lens[label] < per_task_exemplar:
            #                     ssul_exemplar_idx_cls[label].append(idx)
            #                     idxs.append(idx)
            #                     lens[label] += 1
            #             for label in labels_old:
            #                 if lens[label - 1] < per_task_exemplar:
            #                     fg = 0
            #             if fg == 1:
            #                 break
            #
            #         torch.save(ssul_exemplar_idx_cls, ssul_exemplar_path)


            # take index of images with at least one class in labels and all classes in labels+labels_old+[0]


            if idxs_path is not None and os.path.exists(idxs_path):
                idx = np.load(idxs_path).tolist()
            else:
                idx = filter_images(full_voc, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idx, dtype=int))



            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[255] = masking_value

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                if data_masking == "current":
                    tmp_labels = self.labels + [255]
                elif data_masking == "current+old":
                    tmp_labels = labels_old + self.labels + [255]
                elif data_masking == "all":
                    raise NotImplementedError(
                        f"data_masking={data_masking} not yet implemented sorry not sorry."
                    )
                elif data_masking == "new":
                    tmp_labels = self.labels
                    masking_value = 255

                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                )
            else:
                assert False
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_voc, idxs, transform, target_transform)
        else:
            self.dataset = full_voc




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def viz_getter(self, index):
        return self.dataset.viz_getter(index)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
