import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset_nii import BaseVolumeDataset


class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class BrainVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class HippoVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 486420.2188)
        self.target_spacing = (0.125, 0.125, 0.125)
        self.global_mean = 21508.0996
        self.global_std = 60030.5781
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class HepaticVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-726, 3071.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 84.7276
        self.global_std = 40.0027
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-39, 204)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1


class LungVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-939.75, 576.75)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -167.6621
        self.global_std = 367.4284
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class Lung2VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 1749)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 984.0138
        self.global_std = 222.9426
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class Lung3VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 1175.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 815.7423
        self.global_std = 271.6197
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1


class Kits23VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1022 - 30, 3071. + 30)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 61.3800
        self.global_std = 55.3551
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "lung": LungVolumeDataset,
    "lung2": Lung2VolumeDataset,
    'lung3': Lung3VolumeDataset,
    'brain': BrainVolumeDataset,
    'hippo': HippoVolumeDataset,
    'hepatic': HepaticVolumeDataset,
    "kits23": Kits23VolumeDataset
}





def load_data_volume(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]

    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader



def build_dataset(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]

    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    
    return dataset