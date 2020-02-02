
import os
import xml.etree.ElementTree as ET
import glob

from torch.utils.data import Dataset
from PIL import Image

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
        tree = ET.parse(source=self.anno_list_path[idx])
        objs = tree.findall('object')

        boxes = list()
        labels = list()

        for obj in objs:
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

    def __len__(self):
        return len(self.img_list_paths)