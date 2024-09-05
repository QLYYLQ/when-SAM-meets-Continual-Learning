import os

import torch.utils.data as data
from .register import register_training_dataset, register_validation_dataset
from .base_dataset import BaseSegmentation,BaseIncrement
from typing_extensions import override



@register_training_dataset
class Segmentation(BaseSegmentation):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @override
    def _get_path(self):
        # train 和 train_aug 文件一样
        return os.path.join(self.root, "splits","train.txt")

    @override
    def _load_data_path_to_list(self, path):
        images = []
        with open(path, 'r') as f:
            for line in f:
                x = line.strip().split(" ")
                images.append((os.path.join(self.root, x[0][1:]), os.path.join(self.root, x[1][1:])))
        return images


# @register_training_dataset
# class Increment(data.Dataset):
#
#     def __init__(
#             self,
#             root,
#             transform=None,
#             order=None,
#             labels=None,
#             labels_old=None,
#             idxs_path=None,
#             save_path = None,
#             masking=True,
#             overlap=True,
#             data_masking="current",
#             no_memory = True,
#             need_index_name=False,
#             **kwargs
#     ):
#
#         full_voc = Segmentation(root, is_aug=True, transform=None,need_index_name=need_index_name)
#         self.order = order
#         self.labels = []
#         self.labels_old = []
#         self.no_memory = no_memory
#         if labels is not None:
#             # store the labels
#             labels_old = labels_old if labels_old is not None else []
#             # 去除labels中的0号
#             self.__strip_zero(labels)
#             self.__strip_zero(labels_old)
#
#             assert not any(
#                 l in labels_old for l in labels
#             ), "labels and labels_old must be disjoint sets"
#             if not self.no_memory:
#                 # 实现记忆工作
#                 # 创建一个列表，记录了需要使用的过往训练过的图片和target的地址，通过一些策略把这些地址插入到self.dataset.images这个列表中
#                 # 即可，或是在getitem方法中实现选择读入
#                 pass
#             # 下面这一坨注释是：确保了在学习新任务时，模型能够保留足够的旧任务示例，从而在持续学习过程中保持对旧任务的记忆。它通过平衡每个类别的
#             # 示例数量来防止灾难性遗忘，同时也考虑了数据集之间可能存在的重叠，如果用sam的话我们应该不需要这个流程
#
#             # if ssul_exemplar_path is not None:
#             #     print(ssul_exemplar_path)
#             #     if os.path.exists(ssul_exemplar_path):
#             #         ssul_exemplar_idx_cls = torch.load(ssul_exemplar_path)
#             #         for k in ssul_exemplar_idx_cls:
#             #             if opts.method != 'SATS':
#             #                 idxs += ssul_exemplar_idx_cls[k][:7]
#             #             else:
#             #                 idxs += ssul_exemplar_idx_cls[k]
#             #         print('length of ssul-m balanced exemplar samples:', len(idxs))
#             #     if len(idxs) == 0 or not os.path.exists(ssul_exemplar_path) and False:
#             #         print(f'current task:{labels} ssul building exemplar set!')
#             #         per_task_exemplar = opts.ssul_m_exemplar_total / len(labels_old)
#             #         print(f'every class {per_task_exemplar} samples')
#             #         assert idxs_path is not None
#             #         idxs_old = np.load(idxs_path).tolist()
#             #
#             # old_exemplar = (f'./balance_step_exemplar/{opts.dataset}_{opts.task}_step_{opts.step - 1}_exemplar'
#             # f'.pth') ssul_exemplar_idx_cls = torch.load(old_exemplar) lens = {} for label in labels_old:
#             # ssul_exemplar_idx_cls[label - 1] = [] lens[label - 1] = 0 for idx in tqdm(idxs_old): img_cls =
#             # np.unique(np.array(full_voc[idx][1])) fg = 1 print(lens) for label in img_cls:  #QUESTION??? NOT EVERY
#             # CLASS TO BE 20 SAMPLES if label in ssul_exemplar_idx_cls.keys() and lens[label] < per_task_exemplar:
#             # ssul_exemplar_idx_cls[label].append(idx) idxs.append(idx) lens[label] += 1 for label in labels_old: if
#             # lens[label - 1] < per_task_exemplar: fg = 0 if fg == 1: break
#             #
#             #         torch.save(ssul_exemplar_idx_cls, ssul_exemplar_path)
#
#             # take index of images with at least one class in labels and all classes in labels+labels_old+[0]
#
#             if idxs_path is not None and os.path.exists(idxs_path):
#                 idx = load_list_from_path(idxs_path)
#             else:
#                 idx = filter_images(full_voc, labels, labels_old, overlap=overlap)
#                 if save_path is not None:
#                     save_list_from_filter(idx, save_path)
#
#             #if train:
#             #    masking_value = 0
#             #else:
#             #    masking_value = 255
#
#             #self.inverted_order = {label: self.order.index(label) for label in self.order}
#             #self.inverted_order[255] = masking_value
#             # 只在训练
#             masking_value = 255  # Future classes will be considered as background.
#             self.inverted_order = {label: self.order.index(label) for label in self.order}
#             self.inverted_order[255] = 255
#
#             reorder_transform = tv.transforms.Lambda(
#                 lambda t: t.apply_(
#                     lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
#                 )
#             )
#
#             if masking:
#                 if data_masking == "current":
#                     tmp_labels = self.labels + [255]
#                 elif data_masking == "current+old":
#                     tmp_labels = labels_old + self.labels + [255]
#                 elif data_masking == "all":
#                     raise NotImplementedError(
#                         f"data_masking={data_masking} not yet implemented sorry not sorry."
#                     )
#                 elif data_masking == "new":
#                     tmp_labels = self.labels
#                     masking_value = 255
#
#                 target_transform = tv.transforms.Lambda(
#                     lambda t: t.
#                     apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
#                 )
#             else:
#                 target_transform = reorder_transform
#                 assert False
#
#             # make the subset of the dataset
#             self.dataset = full_voc
#             self.dataset.images = idx
#             self.dataset.target_transform = target_transform
#         else:
#             self.dataset = full_voc
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """
#
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     @staticmethod
#     def __strip_zero(labels):
#         while 0 in labels:
#             labels.remove(0)

@register_training_dataset
class Increment(BaseIncrement):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)




@register_validation_dataset
class Validation(data.Dataset):
    def __init__(
            self,
            root,
            classes = None,
            transform=None,
            target_transform=None,
            order=None,
            labels=None,
            labels_old=None,
            idxs_path=None,
            save_path=None,
            masking=True,
            overlap=True,
            data_masking="current",
            **kwargs
    ):
        """感觉测试数据中的实现不需要考虑是否overlap，后续包装中可以通过传入的labels调整dataset为disjoint或者overlap"""
        self.root = root
        self.dataset = Segmentation(root,need_index_name=True,transform=transform,classes=classes,target_transform=target_transform)
        self.order = order
        # 感觉这两个参数目前用不到
        self.labels = labels
        self.labels_old = labels_old
        #
        self.save_path = save_path

