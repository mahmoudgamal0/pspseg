import os
import yaml
import time
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from dataset.dataset import SemData
from util.util import AverageMeter, intersectionAndUnion

device = torch.device('cuda:0')
torch.cuda.set_device(0)

KITTI_CFG = yaml.safe_load(open('./config/kitti.yaml', 'r'))
CFG = yaml.safe_load(open('./config/config.yaml', 'r'))
TEST_CFG = CFG['TEST']

color_dict = KITTI_CFG["color_map"]
learning_map = np.unique(list(KITTI_CFG['learning_map'].values()))
nclasses = learning_map.size
names = [KITTI_CFG['labels'][KITTI_CFG['learning_map_inv'][i]] for i in range(nclasses)]


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

logger = get_logger()
logger.info("=> creating model ...")
logger.info("Classes: {}".format(nclasses))


test_data = SemData(split='TEST')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
from model.pspnet import PSPNet
model = PSPNet(layers=50, classes=nclasses, pretrained=False)
# logger.info(model)
model = torch.nn.DataParallel(model.to(device))
cudnn.benchmark = True

model_path = "./accepted/psp_xyr.pth"
logger.info("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'], strict=False)
logger.info("=> loaded checkpoint '{}'".format(model_path))

def test(test_loader, model, classes):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
          input = input.to(device)
          target = target.to(device)
          output = model(input)
          intersection, union, target = intersectionAndUnion(output.cpu().numpy(), target.cpu().numpy(), nclasses, ignore_index=0)
          intersection_meter.update(intersection)
          union_meter.update(union)
          target_meter.update(target)
          accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
          logger.info('Evaluating {0}/{1}, accuracy {2:.4f}.'.format(i + 1, len(test_loader), accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(nclasses):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}. name: {}'.format(i, iou_class[i], accuracy_class[i], names[i]))

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

test(test_loader, model, nclasses)
