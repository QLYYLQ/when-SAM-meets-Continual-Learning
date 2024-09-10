import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset import dataset_entrypoints
# from ..dataset.voc1 import VOCSegmentationIncremental

# print(is_dataset_registered("VOC.Segmentation"))
# dataset = dataset_entrypoints("VOC.Segmentation")(r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\CVPR2021_PLOP\data\PascalVOC12")
# dataset = VOCSegmentationIncremental(r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\CVPR2021_PLOP\data\PascalVOC12")
from dataset.VOC import Split

order = [i for i in range(21)]

dataset = dataset_entrypoints("VOC.Increment")(r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\CVPR2021_PLOP\data"
                                               r"\PascalVOC12", labels=[2, 4], save_old=[3, 7], idx_path=
                                               r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\my_dataset\test\test"
                                               r".txt",
                                               order=order)
