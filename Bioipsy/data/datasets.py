import os,glob
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
import torch.nn.functional as F

class BiopsyDataset(Dataset):
    def __init__(self,data_path,transforms):
        self.paths=data_path
        self.transforms = transforms

        # 获取所有病例的文件名
        self.case_ids=sorted(os.listdir(os.path.join(data_path,"fixed_images")))
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
        moving_image_path=os.path.join(path,'moving_images',case_id)
        fixed_image_path=os.path.join(path,'fixed_images',case_id)
        moving_label_path=os.path.join(path,'moving_labels',case_id)
        fixed_label_path=os.path.join(path,'fixed_labels',case_id)
        moving_lesion_path=os.path.join(path,'moving_lesion',case_id)
        fixed_lesion_path=os.path.join(path,'fixed_lesion',case_id)

        # 读取 NIfTI 文件
        moving_image=nib.load(moving_image_path).get_fdata()
        fixed_image=nib.load(fixed_image_path).get_fdata()
        moving_label=nib.load(moving_label_path).get_fdata()
        fixed_label=nib.load(fixed_label_path).get_fdata()
        moving_lesion=nib.load(moving_lesion_path).get_fdata()
        fixed_lesion=nib.load(fixed_lesion_path).get_fdata()
        # print(np.unique(moving_label))
        # 数据预处理（可选）
        moving_image = moving_image / np.max(moving_image)  # 标准化到 [0, 1]
        fixed_image = fixed_image / np.max(fixed_image)

        # print(np.unique(moving_image))
        # print(type(moving_image))

        moving_image, fixed_image=moving_image[None,...],fixed_image[None,...]
        moving_label, fixed_label=moving_label[None,...],fixed_label[None,...]
        moving_lesion, fixed_lesion=moving_lesion[None,...],fixed_lesion[None,...]

        # 数据增强

        moving_image,fixed_image = self.transforms([moving_image, fixed_image])
        # 让数据在内存上连续
        moving_image = np.ascontiguousarray(moving_image)
        moving_label = np.ascontiguousarray(moving_label)
        moving_lesion = np.ascontiguousarray(moving_lesion)
        fixed_image = np.ascontiguousarray(fixed_image)
        fixed_label = np.ascontiguousarray(fixed_label)
        fixed_lesion = np.ascontiguousarray(fixed_lesion)

        # print(f'moving.type:{moving_image.dtype}')
        # print(fixed_image.dtype)
        # 将数据转换为 torch.Tensor
        moving_image = torch.from_numpy(moving_image).float()
        moving_label = torch.from_numpy(moving_label).long()
        moving_lesion = torch.from_numpy(moving_lesion).long()
        fixed_image = torch.from_numpy(fixed_image).float()
        fixed_label = torch.from_numpy(fixed_label).long()
        fixed_lesion = torch.from_numpy(fixed_lesion).long()

        # padding image size变为1,128，128，32, uncrop
        # moving_image=F.pad(moving_image,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_image=F.pad(fixed_image,(1,1,0,0,0,0)).unsqueeze(0)
        # moving_label=F.pad(moving_label,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_label=F.pad(fixed_label,(1,1,0,0,0,0)).unsqueeze(0)
        # moving_lesion=F.pad(moving_lesion,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_lesion=F.pad(fixed_lesion,(1,1,0,0,0,0)).unsqueeze(0)
        # print(np.unique(moving_label))

        moving_image=F.pad(moving_image,(1,1,0,0,0,0))
        fixed_image=F.pad(fixed_image,(1,1,0,0,0,0))
        moving_label=F.pad(moving_label,(1,1,0,0,0,0))
        fixed_label=F.pad(fixed_label,(1,1,0,0,0,0))
        moving_lesion=F.pad(moving_lesion,(1,1,0,0,0,0))
        fixed_lesion=F.pad(fixed_lesion,(1,1,0,0,0,0))

        # crop dataset use this
        # moving_image=moving_image[None,...]
        # fixed_image=fixed_image[None,...]
        # moving_label=moving_label[None,...]
        # fixed_label=fixed_label[None,...]
        # moving_lesion=moving_lesion[None,...]
        # fixed_lesion=fixed_lesion[None,...]

        return moving_image,fixed_image,moving_label,fixed_label,moving_lesion,fixed_lesion

    def __len__(self):
        return len(self.case_ids)

class BiopsyInferDataset(Dataset):
    def __init__(self,data_path,transforms):
        self.paths=data_path
        self.transforms = transforms

        # 获取所有病例的文件名
        self.case_ids=sorted(os.listdir(os.path.join(data_path,"fixed_images")))
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
        moving_image_path=os.path.join(path,'moving_images',case_id)
        fixed_image_path=os.path.join(path,'fixed_images',case_id)
        moving_label_path=os.path.join(path,'moving_labels',case_id)
        fixed_label_path=os.path.join(path,'fixed_labels',case_id)
        moving_lesion_path=os.path.join(path,'moving_lesion',case_id)
        fixed_lesion_path=os.path.join(path,'fixed_lesion',case_id)

        # 读取 NIfTI 文件
        moving_image=nib.load(moving_image_path).get_fdata()
        fixed_image=nib.load(fixed_image_path).get_fdata()
        moving_label=nib.load(moving_label_path).get_fdata()
        fixed_label=nib.load(fixed_label_path).get_fdata()
        moving_lesion=nib.load(moving_lesion_path).get_fdata()
        fixed_lesion=nib.load(fixed_lesion_path).get_fdata()

        # 数据预处理（可选）
        moving_image = moving_image / np.max(moving_image)  # 标准化到 [0, 1]
        fixed_image = fixed_image / np.max(fixed_image)

        # print(np.unique(moving_label))
        moving_image, fixed_image=moving_image[None,...],fixed_image[None,...]
        moving_label, fixed_label=moving_label[None,...],fixed_label[None,...]
        moving_lesion, fixed_lesion=moving_lesion[None,...],fixed_lesion[None,...]
        # 数据增强

        moving_image,moving_label = self.transforms([moving_image,moving_label])
        fixed_image,fixed_label = self.transforms([fixed_image,fixed_label])
        # 让数据在内存上连续
        moving_image = np.ascontiguousarray(moving_image)
        moving_label = np.ascontiguousarray(moving_label)
        moving_lesion = np.ascontiguousarray(moving_lesion)
        fixed_image = np.ascontiguousarray(fixed_image)
        fixed_label = np.ascontiguousarray(fixed_label)
        fixed_lesion = np.ascontiguousarray(fixed_lesion)

        # print(f'moving.type:{moving_label.dtype}')
        # print(fixed_image.dtype)
        # 将数据转换为 torch.Tensor
        moving_image = torch.from_numpy(moving_image).float()
        moving_label = torch.from_numpy(moving_label).long()
        moving_lesion = torch.from_numpy(moving_lesion).long()
        fixed_image = torch.from_numpy(fixed_image).float()
        fixed_label = torch.from_numpy(fixed_label).long()
        fixed_lesion = torch.from_numpy(fixed_lesion).long()

        # padding image size变为1,128，128，32, uncrop
        # moving_image=F.pad(moving_image,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_image=F.pad(fixed_image,(1,1,0,0,0,0)).unsqueeze(0)
        # moving_label=F.pad(moving_label,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_label=F.pad(fixed_label,(1,1,0,0,0,0)).unsqueeze(0)
        # moving_lesion=F.pad(moving_lesion,(1,1,0,0,0,0)).unsqueeze(0)
        # fixed_lesion=F.pad(fixed_lesion,(1,1,0,0,0,0)).unsqueeze(0)
        # print(np.unique(moving_label))

        moving_image=F.pad(moving_image,(1,1,0,0,0,0))
        fixed_image=F.pad(fixed_image,(1,1,0,0,0,0))
        moving_label=F.pad(moving_label,(1,1,0,0,0,0))
        fixed_label=F.pad(fixed_label,(1,1,0,0,0,0))
        moving_lesion=F.pad(moving_lesion,(1,1,0,0,0,0))
        fixed_lesion=F.pad(fixed_lesion,(1,1,0,0,0,0))

        # crop dataset use this
        # moving_image=moving_image[None,...]
        # fixed_image=fixed_image[None,...]
        # moving_label=moving_label[None,...]
        # fixed_label=fixed_label[None,...]
        # moving_lesion=moving_lesion[None,...]
        # fixed_lesion=fixed_lesion[None,...]




        # 数据预处理或增强
        # if self.transforms:
        #     moving_image = self.transforms(moving_image)
        #     fixed_image = self.transforms(fixed_image)


        return moving_image,fixed_image,moving_label,fixed_label,moving_lesion,fixed_lesion

    def __len__(self):
        return len(self.case_ids)