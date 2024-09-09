from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """这个类被设计出来是基于持续学习中需要调整数据集这一需求，第一版比较粗糙暴力。实现办法是dataset被dataloader封装以后依然可以通过.dataset访问"""
    def __init__(self, dataset_list):
        super(BaseDataset, self).__init__()
        for index, dataset in enumerate(dataset_list):
            setattr(self, f"dataset_{index}", dataset)
        self.stage = 0
        self.max_stage = len(dataset_list)

    def _check_stage(self):
        if self.stage >= self.max_stage:
            raise AttributeError("the stage is larger than the maximum stage")
        return True

    def __getitem__(self, index):
        pass











# class test(Dataset):
#     def __init__(self):
#         super(test, self).__init__()
#         self.stage = 999
#     def __getitem__(self, index):
#         return index,self.stage
#
#     def __len__(self):
#         return 100000
#
#
# if __name__ == '__main__':
#     dataset = test()
#     dataloader = DataLoader(dataset,batch_size=10)
#     for batch in dataloader:
#         print(batch)
#         dataloader.dataset.stage = 1