import os
import numpy as np
from dataset.dataloader import Dataloader
from utils import get_dataset_config
from utils.create_dataset import load_dataset_from_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from utils.finetune_sam.create_finetune_sam_point_and_bbox import (find_separated_boundaries,
                                                                   create_bbox_from_origin_and_separated_label,
                                                                   create_point_from_origin_and_separated_label,
                                                                   create_point_label_from_origin_and_separated_points
                                                                   )
from utils.finetune_sam.show_image import show_image_with_mask_and_prompt
config = get_dataset_config("ADE")
print(config)
dataset,val_dataset = load_dataset_from_config(config, 2, None)
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
        # label = label.numpy()
        # a = label[0].squeeze()
        label_number = data["label_index"]
        a, b = find_separated_boundaries(label, label_number)
        bbox1,bbox2 = create_bbox_from_origin_and_separated_label(label,b)
        # show_image_with_mask_and_prompt(image[0], b[0],"test",None,bboxs = bbox2[0])
        # show_image_with_mask_and_prompt(image[0],label[0],"test_1",None,bboxs = bbox1[0])
        points11,points12,points21,points22 = create_point_from_origin_and_separated_label(label,b,"number",background_number=2,foreground_number=2)
        point_label11,point_label12,point_label21,point_label22 = create_point_label_from_origin_and_separated_points(points11,points21)
        # print(points11,points12)
        # print()
        # bbox1[0] = bbox1[0][0:1,:]
        # label_value = np.unique(label[0])
        # label_value = [x for x in label_value if x != 0]
        # label[0][label[0]!=label_value[0]]=0
        show_image_with_mask_and_prompt(image[0], label[0], "test", points11[0], points12[0], bboxs=bbox1[0],
                                        type="origin")
        show_image_with_mask_and_prompt(image[0], b[0], "test_1", points21[0], points22[0], bboxs=bbox2[0],
                                        type="separated")
        print("finish")
        # out = vit(image)
        #masks = BaseModel.align_mask_dimension(out,mask)
        #print(masks)
    dataset.update_stage(1)
    dataloader = Dataloader(dataset, batch_size=5)
