from PIL import Image, ImageFilter
from glob import glob
import os
from torch.utils.data import Dataset
import math
import numpy as np
import shutil
import random


random.seed(128)


data_dir = 'data/imagenet/train/*/*.JPEG'
data_dir = os.path.join(data_dir)
image_paths = sorted(glob(data_dir))

selected_path = random.sample(image_paths, k=50)

print(selected_path[0])

for path in selected_path:
    shutil.copy(path, 'imagenet_subset')