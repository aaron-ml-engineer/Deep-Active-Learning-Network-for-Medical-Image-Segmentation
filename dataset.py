import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, models, transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
       
        np_img = np.load(img_path)
        np_mask = np.load(mask_path)
        np_img = np_img.transpose((2, 0, 1))

        WT_Label = np_mask.copy()
        WT_Label[np_mask == 1] = 1.
        WT_Label[np_mask == 2] = 1.
        WT_Label[np_mask == 4] = 1.
        TC_Label = np_mask.copy()
        TC_Label[np_mask == 1] = 1.
        TC_Label[np_mask == 2] = 0.
        TC_Label[np_mask == 4] = 1.
        ET_Label = np_mask.copy()
        ET_Label[np_mask == 1] = 0.
        ET_Label[np_mask == 2] = 0.
        ET_Label[np_mask == 4] = 1.
        np_label = np.empty((160, 160, 3))
        np_label[:, :, 0] = WT_Label
        np_label[:, :, 1] = TC_Label
        np_label[:, :, 2] = ET_Label
        np_label = np_label.transpose((2, 0, 1))

        np_label = np_label.astype("float32")
        np_img = np_img.astype("float32")

        return np_img, np_label

