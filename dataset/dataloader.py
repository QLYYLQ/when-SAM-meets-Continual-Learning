from torch.utils.data import DataLoader as PytorchDataLoader


class Dataloader(PytorchDataLoader):
    def __init__(self, dataset, batch_size=1):
        collate_fn = self.collate_fn
        super(Dataloader, self).__init__(dataset, batch_size, collate_fn=collate_fn)

    @staticmethod
    def collate_fn(batch):
        data = [item["data"][0] for item in batch]
        label = [item["data"][1] for item in batch]
        text_prompt = [item["text_prompt"] for item in batch]
        return data, label, text_prompt
