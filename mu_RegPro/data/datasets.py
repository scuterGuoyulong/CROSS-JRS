import os,glob
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
import torch.nn.functional as F
# from torchio.transforms import (
#     RescaleIntensity,
#     RandomAffine,
#     RandomElasticDeformation,
#     Compose,
#     OneOf,
#     Resample,
#     RandomFlip,
#     CropOrPad,
#     Lambda
# )
class muProRegDataset(Dataset):
    def __init__(self,data_path,transforms):
        self.paths=data_path
        self.transforms = transforms

        # 获取所有病例的文件名
        self.case_ids=sorted(os.listdir(os.path.join(data_path,"us_images")))

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self,index):
        """
        根据索引返回 MR 和 US 图像及其对应的标签。
        Args:
            index (int): 数据索引。
        Returns:
            dict: 包含图像和标签的字典，格式为：
                {
                    'mr_image': torch.Tensor,
                    'mr_label': torch.Tensor,
                    'us_image': torch.Tensor,
                    'us_label': torch.Tensor
                }
        """
        path=self.paths
        # 获取当前病例的文件名
        case_id=self.case_ids[index]

        # 构建各路径
        moving_image_path=os.path.join(path,'mr_images',case_id)
        fixed_image_path=os.path.join(path,'us_images',case_id)
        moving_label_path=os.path.join(path,'mr_labels',case_id)
        fixed_label_path=os.path.join(path,'us_labels',case_id)

        # 读取 NIfTI 文件
        moving_image=nib.load(moving_image_path).get_fdata()
        fixed_image=nib.load(fixed_image_path).get_fdata()
        moving_label=nib.load(moving_label_path).get_fdata()
        fixed_label=nib.load(fixed_label_path).get_fdata()

        # print(np.unique(moving_label))
        # 数据预处理（可选）
        moving_image = (moving_image-np.min(moving_image)) / (np.max(moving_image)-np.min(moving_image) ) # 标准化到 [0, 1]
        fixed_image = (fixed_image-np.min(fixed_image)) / (np.max(fixed_image)-np.min(fixed_image) ) # 标准化到 [0, 1]

        # print(np.unique(moving_image))
        # print(type(moving_image))

        moving_image, fixed_image=moving_image[None,...],fixed_image[None,...]
        moving_label, fixed_label=moving_label[None,...],fixed_label[None,...]

        # 数据增强

        moving_image,fixed_image = self.transforms([moving_image, fixed_image])
        # 让数据在内存上连续
        moving_image = np.ascontiguousarray(moving_image)
        moving_label = np.ascontiguousarray(moving_label)
        fixed_image = np.ascontiguousarray(fixed_image)
        fixed_label = np.ascontiguousarray(fixed_label)

        # print(f'moving.type:{moving_image.dtype}')
        # print(fixed_image.dtype)
        # 将数据转换为 torch.Tensor
        moving_image = torch.from_numpy(moving_image).float()
        moving_label = torch.from_numpy(moving_label).long()
        fixed_image = torch.from_numpy(fixed_image).float()
        fixed_label = torch.from_numpy(fixed_label).long()

        return moving_image,fixed_image,moving_label,fixed_label

    def __len__(self):
        return len(self.case_ids)

    def create_transforms(self):
        transforms = []

        # clipping to remove outliers (if any)
        # clip_intensity = Lambda(VolumeDataset.clip_image, types_to_apply=[torchio.INTENSITY])
        # transforms.append(clip_intensity)

        rescale = RescaleIntensity((-1, 1))
        # normalize with mu = 0 and sigma = 1/3 to have data in -1...1 almost
        # ZNormalization()

        # transforms.append(rescale)

        # if self.mode == 'train':
        #     # transforms = [rescale]
        #     if 'affine' in self.opt.transforms:
        #         transforms.append(RandomAffine(translation=5, p=0.8))
        #
        #     if 'flip' in self.opt.transforms:
        #         transforms.append(RandomFlip(axes=(0, 2), p=0.8))


        spatial = OneOf(
            {RandomAffine(translation=5): 0.8, RandomElasticDeformation(): 0.2},
            p=0.75,
        )
        transforms += [RandomFlip(axes=(0, 2), p=0.25), spatial]
            # transforms += [RandomBiasField()]

        transforms.append(CropOrPad((80,80,80), padding_mode='minimum'))
        transform = Compose(transforms)

        self.denoising_transform = None
        if len(self.opt.denoising) > 0:
            if 'median' in self.opt.denoising:
                self.denoising_transform = Lambda(VolumeDataset.median_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])
            if 'lee_filter' in self.opt.denoising:
                self.denoising_transform = Lambda(lee_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])

        self.zoom_transform = Compose([Resample(0.5), CropOrPad(self.opt.origshape, padding_mode='minimum')])

        return transform

class muProRegInferDataset(Dataset):
    def __init__(self,data_path,transforms):
        self.paths=data_path
        self.transforms = transforms

        # 获取所有病例的文件名
        self.case_ids=sorted(os.listdir(os.path.join(data_path,"us_images")))

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self,index):
        """
        根据索引返回 MR 和 US 图像及其对应的标签。
        Args:
            index (int): 数据索引。
        Returns:
            dict: 包含图像和标签的字典，格式为：
                {
                    'mr_image': torch.Tensor,
                    'mr_label': torch.Tensor,
                    'us_image': torch.Tensor,
                    'us_label': torch.Tensor
                }
        """
        path=self.paths
        # 获取当前病例的文件名
        case_id=self.case_ids[index]

        # 构建各路径
        moving_image_path=os.path.join(path,'mr_images',case_id)
        fixed_image_path=os.path.join(path,'us_images',case_id)
        moving_label_path=os.path.join(path,'mr_labels',case_id)
        fixed_label_path=os.path.join(path,'us_labels',case_id)

        # 读取 NIfTI 文件
        moving_image=nib.load(moving_image_path).get_fdata()
        fixed_image=nib.load(fixed_image_path).get_fdata()
        moving_label=nib.load(moving_label_path).get_fdata()
        fixed_label=nib.load(fixed_label_path).get_fdata()

        # 数据预处理（可选）
        moving_image = (moving_image-np.min(moving_image)) / (np.max(moving_image)-np.min(moving_image) ) # 标准化到 [0, 1]
        fixed_image = (fixed_image-np.min(fixed_image)) / (np.max(fixed_image)-np.min(fixed_image) ) # 标准化到 [0, 1]


        # print(np.unique(moving_label))
        moving_image, fixed_image=moving_image[None,...],fixed_image[None,...]
        moving_label, fixed_label=moving_label[None,...],fixed_label[None,...]
        # 数据增强

        moving_image,moving_label = self.transforms([moving_image,moving_label])
        fixed_image,fixed_label = self.transforms([fixed_image,fixed_label])
        # 让数据在内存上连续
        moving_image = np.ascontiguousarray(moving_image)
        moving_label = np.ascontiguousarray(moving_label)
        fixed_image = np.ascontiguousarray(fixed_image)
        fixed_label = np.ascontiguousarray(fixed_label)

        # print(f'moving.type:{moving_label.dtype}')
        # print(fixed_image.dtype)
        # 将数据转换为 torch.Tensor
        moving_image = torch.from_numpy(moving_image).float()
        moving_label = torch.from_numpy(moving_label).long()
        fixed_image = torch.from_numpy(fixed_image).float()
        fixed_label = torch.from_numpy(fixed_label).long()

        return moving_image,fixed_image,moving_label,fixed_label

    def __len__(self):
        return len(self.case_ids)