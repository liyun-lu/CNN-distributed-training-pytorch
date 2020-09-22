import h5py

import torch
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MOFsDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, transform=None, target_transform=None, data_size=0, file_path=None):
        imgs = []

        f = h5py.File(file_path, 'r')
        inputs = f['inputs'][:data_size]     # all data
        lables = f['lables'][:data_size]
        for i in range(data_size):
            x = inputs[i]
            y = lables[i]
            # inputs.dtype=int
            x = np.array(x, np.float)
            y = np.array(y, np.float)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            x = x.float()
            y = y.float()
            x = x.to(device)
            y = y.to(device)
            # print(i, x, y)
            imgs.append((x, y))
        f.close()

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.imgs[index]

        # print(index, img)
        x = img[0]
        y = img[1]
        return x, y

    def __len__(self):
        return len(self.imgs)









