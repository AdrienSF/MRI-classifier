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








class DecathlonDataset(Randomizable, CacheDataset):
    """
    The Dataset to automatically download the data of Medical Segmentation Decathlon challenge
    (http://medicaldecathlon.com/) and generate items for training, validation or test.
    It will also load these properties from the JSON config file of dataset. user can call `get_properties()`
    to get specified properties or all the properties loaded.
    It's based on :py:class:`monai.data.CacheDataset` to accelerate the training process.

    Args:
        root_dir: user's local directory for caching and loading the MSD datasets.
        task: which task to download and execute: one of list ("Task01_BrainTumour", "Task02_Heart",
            "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas",
            "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon").
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
            for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        download: whether to download and extract the Decathlon from resource link, default is False.
            if expected file already exists, skip downloading even set it to True.
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.2.
            user can manually copy tar file or dataset folder to the root directory.
        seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
            note to set same seed for `training` and `validation` sections.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.

    Raises:
        ValueError: When ``root_dir`` is not a directory.
        ValueError: When ``task`` is not one of ["Task01_BrainTumour", "Task02_Heart",
            "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas",
            "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"].
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not selected (``download=False``).

    Example::

        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_data = DecathlonDataset(
            root_dir="./", task="Task09_Spleen", transform=transform, section="validation", seed=12345, download=True
        )

        print(val_data[0]["image"], val_data[0]["label"])

    """

    resource = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
        "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
        "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
    }
    md5 = {
        "Task01_BrainTumour": "240a19d752f0d9e9101544901065d872",
        "Task02_Heart": "06ee59366e1e5124267b774dbd654057",
        "Task03_Liver": "a90ec6c4aa7f6a3d087205e23d4e6397",
        "Task04_Hippocampus": "9d24dba78a72977dbd1d2e110310f31b",
        "Task05_Prostate": "35138f08b1efaef89d7424d2bcc928db",
        "Task06_Lung": "8afd997733c7fc0432f71255ba4e52dc",
        "Task07_Pancreas": "4f7080cfca169fa8066d17ce6eb061e4",
        "Task08_HepaticVessel": "641d79e80ec66453921d997fbf12a29c",
        "Task09_Spleen": "410d4a301da4e5b2f6f86ec3ddba524e",
        "Task10_Colon": "bad7a188931dc2f6acf72b08eb6202d0",
    }

    def __init__(
        self,
        root_dir: str,
        task: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)
        if task not in self.resource:
            raise ValueError(f"Unsupported task: {task}, available options are: {list(self.resource.keys())}.")
        dataset_dir = os.path.join(root_dir, task)
        tarfile_name = f"{dataset_dir}.tar"
        if download:
            download_and_extract(self.resource[task], tarfile_name, root_dir, self.md5[task])

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it."
            )
        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(dataset_dir)
        transform(data)
        # as `release` key has typo in Task04 config file, ignore it.
        property_keys = [
            "name",
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "labels",
            "numTraining",
            "numTest",
        ]
        self._properties = load_decathlon_properties(os.path.join(dataset_dir, "dataset.json"), property_keys)
        if transform == ():
            transform = LoadImaged(["image", "label"])
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices


    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)


    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        return {}


    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        section = "training" if self.section in ["training", "validation"] else "test"
        datalist = load_decathlon_datalist(os.path.join(dataset_dir, "dataset.json"), True, section)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        if self.section == "test":
            return datalist
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        if self.section == "training":
            self.indices = indices[val_length:]
        else:
            self.indices = indices[:val_length]

        return [datalist[i] for i in self.indices]






























def _generate_data_list(dataset_dir: str):
    section = "training" #if section in ["training", "validation"] else "test"
    datalist = load_decathlon_datalist(os.path.join(dataset_dir, "dataset.json"), True, section)
    return _split_datalist(datalist)


def _split_datalist(datalist):
    length = len(datalist)
    indices = np.arange(length)
    # np.random.shuffle(indices)

    val_length = int(length * .2)
    indices = indices[val_length:]

    return [datalist[i] for i in indices]


# load = LoadImaged(["image", "label"])

# print(_generate_data_list("cn_mini/Task01_BrainTumour"))

# load(_generate_data_list("cn_mini/Task01_BrainTumour"))


train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"])
    ])



train_ds = DecathlonDataset(
    root_dir="cn_mini",
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)


data_dict = [{'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_1_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_1_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_2_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_2_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_4_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_4_seg.nii.gz'}, {'image': 'cn_mini/Task01_BrainTumour/imagesTr/cn_5_orig.nii.gz', 'label': 'cn_mini/Task01_BrainTumour/labelsTr/cn_5_seg.nii.gz'}]
train_transform(data_dict)