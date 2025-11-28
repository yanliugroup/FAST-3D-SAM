import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset
from .base_dataset_nii import BaseVolumeDataset as BaseVolumeDataset_nii
from torch.utils.data import ConcatDataset

# use all pixel to calculate mean and std
dataset_stats = {'lung': {'mean': -322.5155297041428, 'std': 469.8601382457649, 'min': -1024.0, 'max': 19315.0}, 'lung2': {'mean': 674.6135723435852, 'std': 528.6858683312973, 'min': -10748.0, 'max': 20339.0}, 'Lung421': {'mean': 388.9518236165722, 'std': 465.7339401348388, 'min': -1024.0, 'max': 4095.0}, 'pancreas': {'mean': -125.650033914228, 'std': 359.4353566304122, 'min': -2048.0, 'max': 4009.0}, 'kits23': {'mean': -139.8815115572298, 'std': 362.21033950802564, 'min': -6986.0, 'max': 18326.0}, 'hepatic': {'mean': -97.51201450512347, 'std': 340.1144827002664, 'min': -1024.0, 'max': 3072.0}}

# use foreground pixel to calculate mean and std
{'lung': {'mean': -158.40746487386428, 'std': 324.82639474513553, 'min': -1393.5, 'max': 3040.5}, 'lung2': {'mean': 843.4502890881769, 'std': 332.2021557825976, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 681.1772523003802, 'std': 502.2254769460235, 'min': -1509.4, 'max': 4315.4}, 'pancreas': {'mean': 79.326213811815, 'std': 78.33553180106968, 'min': -1404.9, 'max': 3477.9}, 'kits23': {'mean': 114.43007901992068, 'std': 75.53780947379572, 'min': -1432.4, 'max': 3480.4}, 'hepatic': {'mean': 163.68194368793922, 'std': 39.023300794136574, 'min': -983.7, 'max': 3440.7}}

# change upper bound and lower bound
{'lung': {'mean': -158.40746487386428, 'std': 324.82639474513553, 'min': -1024.0, 'max': 3040.5}, 'lung2': {'mean': 843.4502890881769, 'std': 332.2021557825976, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 681.1772523003802, 'std': 502.2254769460235, 'min': -1024.0, 'max': 4095.0}, 'pancreas': {'mean': 79.326213811815, 'std': 78.33553180106968, 'min': -1404.9, 'max': 3477.9}, 'kits23': {'mean': 114.43007901992068, 'std': 75.53780947379572, 'min': -1432.4, 'max': 3480.4}, 'hepatic': {'mean': 163.68194368793922, 'std': 39.023300794136574, 'min': -983.7, 'max': 3072.0}}


class KiTSVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1432.4, 3480.4)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 114.43007901992068
        self.global_std = 75.53780947379572
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class BrainVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class HippoVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 486420.2188)
        self.target_spacing = (0.2, 0.2, 0.2)
        self.global_mean = 21508.0996
        self.global_std = 60030.5781
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1404.9, 3477.9)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 79.326213811815
        self.global_std = 78.33553180106968
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1


# NOTE: 这里如果是128的话，需要是BaseVolumeDataset
class LungVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.0, 3040.5)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -158.40746487386428
        self.global_std = 324.82639474513553
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class Lung2VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-369.5, 4064.5)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 843.4502890881769
        self.global_std = 332.2021557825976
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
        
class LungTestVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-939.75, 576.75)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -167.6621
        self.global_std = 367.4284
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class Lung3VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.0, 4095.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 681.1772523003802
        self.global_std = 502.2254769460235
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
        
class LungHosVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-27, 520)
        #self.intensity_range = (-527, 1020)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 216.8536
        self.global_std = 75.9415
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class Kits23VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1432.4, 3480.4)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 114.43007901992068
        self.global_std = 75.53780947379572
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2

class HepaticVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-983.7, 3072.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 163.68194368793922
        self.global_std = 39.023300794136574
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "lung": LungVolumeDataset,
    "lung2": Lung2VolumeDataset,
    'lung3': Lung3VolumeDataset,
    'Lung421': Lung3VolumeDataset,
    'brain': BrainVolumeDataset,
    'hippo': HippoVolumeDataset,
    "kits23": Kits23VolumeDataset,
    'hepatic': HepaticVolumeDataset,
    'lung_test': LungTestVolumeDataset,
    'lung_hospital': LungHosVolumeDataset,
    'lung_hospital_multi': LungHosVolumeDataset
}


def load_data_volume(
    *,
    datas,
    path_prefixes,
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
    datasets = []
    for data, path_prefix in zip(datas, path_prefixes):
        dataset = build_dataset(
            data=data,
            path_prefix=path_prefix,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
            deterministic=deterministic,
            augmentation=augmentation,
            fold=fold,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop = do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            num_worker=num_worker,
        )
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)

    if deterministic:
        loader = DataLoader(
            concat_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader

def load_data_volume_single(
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

def load_datasets(
    *,
    datas,
    path_prefixes,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(128, 128, 128),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=8,
):
    datasets = []
    for data, path_prefix in zip(datas, path_prefixes):
        dataset = build_dataset(
            data=data,
            path_prefix=path_prefix,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
            deterministic=deterministic,
            augmentation=augmentation,
            fold=fold,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop = do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            num_worker=num_worker,
        )
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)

    return concat_dataset

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