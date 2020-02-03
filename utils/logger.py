import logging
from torchvision.transforms import functional as F

def get_logger(args):
    logger = logging.getLogger(args.type)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(args.save_log_dir + args.type + '_logging.log')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_hander)

    return logger

