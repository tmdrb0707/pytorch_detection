import torch
import xml.etree.ElementTree as ET

import os

from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.img_list_path = [os.path.join(self.img_dir, f.name) for f in os.scandir(img_dir)]
        self.anno_list_path = [os.path.join(self.anno_dir, f.name) for f in os.scandir(anno_dir)]

    def __getitem__(self, idx):
        img = Image.open(self.img_list_path[idx]).convert('RGB')
        target = load_anno(self.anno_list_path[idx])
        target['image_id'] = torch.tensor([idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_list_path)


def load_anno(anno_path):

    tree = ET.parse(source=anno_path)
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    object = root.findall('object')

    boxes = []
    label_numbers = []
    areas = []
    for obj in object:
        label = obj.find('name').text
        label_number = label_map(label)
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        label_numbers.append(label_number)
        areas.append((xmax - xmin) * (ymax - ymin))

    target = {}
    target['width'] = torch.as_tensor(width, dtype=torch.int64)
    target['height'] = torch.as_tensor(height, dtype=torch.int64)
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    target['labels'] = torch.as_tensor(label_numbers, dtype=torch.int64)

    # Mask-R-CNN에 필요한 속성들.
    # Faster-R-CNN에서는 필요없지만 API를 사용하시 위해서 Dummy 생성.
    iscrowd = torch.zeros((len(label_numbers),), dtype=torch.int64)
    target['area'] = torch.as_tensor(areas, dtype=torch.float32)
    target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

    return target


def label_map(label):

    if label == "Car":
        label_number = 1

    return label_number