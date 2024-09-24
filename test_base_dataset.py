import os
from utils import get_dataset_config
from utils.create_dataset import load_dataset_from_config
from dataset.dataloader import Dataloader
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from utils.create_sam_finetune_bbox import find_separated_boundaries,create_bbox
config = get_dataset_config("ADE")
print(config)
dataset,val = load_dataset_from_config(config, 2, None)
dataloader = Dataloader(dataset, batch_size=3)
epoch = 1
# from model.encoder.image_encoder.swin import build_swin
#
# swin = build_swin("swin_T_224_1k")
# from model.encoder.image_encoder.vit import build_vit
# vit = build_vit("vit_b")
flag = True
for i in range(epoch):
    print(f"epoch:{epoch}")
    j = 0
    for data in dataloader:
        image = data["image"]
        mask = data["mask"]
        label = data["label"]
        print("finish")
        label = label.numpy()
        # a = label[0].squeeze()
        label_number = data["label_index"]
        a, b = find_separated_boundaries(label, label_number)

        # out = vit(image)
        #masks = BaseModel.align_mask_dimension(out,mask)
        #print(masks)
    dataset.update_stage(1)
    dataloader = Dataloader(dataset, batch_size=5)
