
import os
import xml.etree.ElementTree as ET
import glob

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, root, anno, transforms=None):
        self.root = root
        self.anno = anno
        self.transforms = transforms
        self.exts = ['jpg', 'png', 'jpeg']

        self.img_list_path = list()
        for f in os.scandir(self.root):
            if f.is_dir():
                raise FileExistsError
            else:
                self.img_list_path.append(os.path.join(self.root, f.name))

        self.anno_list_path = list()
        for f in os.scandir(self.anno):
            if f.is_dir():
                raise FileExistsError
            else:
                self.anno_list_path.append(os.path.join(self.anno, f.name))

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

if __name__ == '__main__':

    a = ['a','b','c']

    b = 'a'

    if a[0] == b:
        raise FileExistsError('iiiii')