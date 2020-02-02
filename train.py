import torch
import torchvision
import time
import argparse
import os
import logging

from torchvision import transforms as T

from customdataset import CustomDataset

# 인자들

parser = argparse.ArgumentParser(description='argument for training')
parser.add_argument('--img_dir', default='./data/img', help='img folder of dataset')
parser.add_argument('--anno_dir', default='./data/anno', help='annotation folder of dataset')
parser.add_argument('--checkpoint_path', default=None, help='path for saving checkpoint')
parser.add_argument('--save_log_dir', default='./log/', help='dir for saving Log')

args = parser.parse_args()


train_logger = logging.getLogger('train_logger')

# file_handler = logging.FileHandler(args.save_log_dir + 'logging.log')
# train_logger.addHandler(file_handler)

train_logger.setLevel(logging.INFO)
train_logger.info('Start Train Log -> Saving Path is {}'.format(args.save_log_dir))

def load_data(img_dir, anno_dir):
    '''
        전체 데이터Set에 대한 경로를 List에 담아 Return한다.
        root : 이미지 데이터가 들어있는 최상위 경로
    :return:
    '''
    transforms = T.Compose([T.Resize((128, 128)),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    CustomDataset(img_dir, anno_dir, transforms=transforms)

def main():

    # 경로 설정 부분
    img_dir = args.img_dir
    anno_dir = args.anno_dir
    os.makedirs(args.checkpoint_path, exist_ok=True)

    train_logger.info('Loading DataSet')
    load_data(img_dir, anno_dir)
    train_logger.info('Done')

if __name__ == '__main__':
    main()