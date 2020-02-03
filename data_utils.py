import torch
import xml.etree.ElementTree as ET

def loadAnns(anno_file):

    tree = ET.parse(source=anno_file)
    root = tree.getroot()
    # folder = root.find('folder').text
    # filename = root.find('filename').text
    # path = root.find('path').text
    # database = root.find('source').find('database').text
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    depth = int(root.find('size').find('depth').text)
    segmented = int(root.find('segmented').text)
    object = root.findall('object')

    names = list()
    boxes = list()
    poses = list()
    truncateds = list()
    difficaults = list()

    for obj in object:
        # name = obj.find('name').text
        # pose = obj.find('pose').text
        truncated = int(obj.find('truncated').text)
        difficault = int(obj.find('difficult').text)
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        # names.append([name])
        # poses.append([pose])
        truncateds.append([truncated])
        difficaults.append([difficault])
        boxes.append([xmin, ymin, xmax, ymax])

    target = dict()
    # target['folder'] = folder
    # target['filename'] = filename
    # target['path'] = path
    # target['database'] = database
    target['width'] = torch.as_tensor(width, dtype=torch.int64)
    target['height'] = torch.as_tensor(height, dtype=torch.int64)
    target['depth'] = torch.as_tensor(depth, dtype=torch.int64)
    target['segmented'] = torch.as_tensor(segmented, dtype=torch.int64)
    # target['name'] = names
    # target['pose'] = poses
    target['truncated'] = torch.as_tensor(truncated, dtype=torch.int64)
    target['difficault'] = torch.as_tensor(difficault, dtype=torch.int64)
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)

    return target


