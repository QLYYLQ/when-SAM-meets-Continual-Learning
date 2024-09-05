import sys
import os.path as osp
from pathlib import Path

root = Path(__file__).resolve().parent.parent.joinpath("data")
print(root)
path = r"/PascalVoc12"
print(root.joinpath(path))