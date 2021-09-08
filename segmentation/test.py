import os, sys
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
# from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, load_decathlon_datalist, CacheDataset, load_decathlon_properties
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Randomizable,
)
from monai.utils import set_determinism

import torch

from typing import Callable, Dict, List, Optional, Sequence, Union



train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"])
        # EnsureChannelFirstd(keys="image")
    ])




data_dict = [{'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_1_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_1_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_2_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_2_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_4_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_4_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_5_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_5_seg.nii.gz'}]
res = train_transform(data_dict)

print(res[0]['image_meta_dict']['original_channel_dim'])