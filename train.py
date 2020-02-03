import torch
import torchvision
import time
import argparse
import os
from customdataset import CustomDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils import transforms as T
from utils.collate_fn import collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils.reduce_dict import reduce_dict

parser = argparse.ArgumentParser(description='argument for training')
parser.add_argument('--type', default='train', help='img folder of dataset')
parser.add_argument('--img_dir', default='.\\data\\img', help='img folder of dataset')
parser.add_argument('--anno_dir', default='.\\data\\anno', help='annotation folder of dataset')
parser.add_argument('--checkpoint_dir', default='./models/', help='path for saving checkpoint')
parser.add_argument('--save_log_dir', default='./log/', help='dir for saving Log')
args = parser.parse_args()


def load_data(img_dir, anno_dir):
    '''
        전체 데이터Set에 대한 경로를 List에 담아 Return한다.
        root : 이미지 데이터가 들어있는 최상위 경로
    :return:
    '''
    transforms = list()
    transforms.append(T.ToTensor())
    transforms = T.Compose(transforms)
    dataset = CustomDataset(img_dir, anno_dir, transforms=transforms)
    return dataset

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 경로 설정 부분
    img_dir = args.img_dir
    anno_dir = args.anno_dir

    # 폴더 생성 부분
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.save_log_dir, exist_ok=True)

    # logging 호출
    train_logger = get_logger(args)

    train_logger.info('Loading DataSet')
    dataset = load_data(img_dir, anno_dir)
    train_logger.info('Done')

    train_loader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for idx, (imgs, targets) in enumerate(train_loader):
        images = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        print(1)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == '__main__':
    main()