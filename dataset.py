
import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(400, 400)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = glob(os.path.join(self.data_path, '*.jpg'))

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        try:
            img_cover = Image.open(img_cover_path).convert('RGB')
            img_cover = ImageOps.fit(img_cover, self.size)
            img_cover = np.array(img_cover, dtype=np.float32) / 255.
        except:
            img_cover = np.zeros((3, self.size[0], self.size[1]), dtype=np.float32)
        
        secret = np.random.binomial(1, 0.5, self.secret_size)

        return img_cover, secret

    def __len__(self):
        return len(self.files_list)
