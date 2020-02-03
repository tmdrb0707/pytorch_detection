import os

from data_utils import loadAnns
from torch.utils.data import Dataset
from PIL import Image
from utils import transforms

class CustomDataset(Dataset):

    def __init__(self, img_dir, anno_dir, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.exts = ['jpg', 'png', 'jpeg', 'bmp']

        self.img_list_path = [os.path.join(self.img_dir, f.name) for f in os.scandir(img_dir)]
        self.anno_list_path = [os.path.join(self.anno_dir, f.name) for f in os.scandir(anno_dir)]

    def __getitem__(self, idx):
        img = Image.open(self.img_list_path[idx]).convert('RGB')
        target = loadAnns(self.anno_list_path[idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_list_path)