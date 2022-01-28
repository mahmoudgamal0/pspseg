import yaml
import numpy as np
import os
import random
import time
import logging

from dataset.dataset import SemData

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU


device = torch.device('cuda:0')
torch.cuda.set_device(0)

KITTI_CFG = yaml.safe_load(open('./config/kitti.yaml', 'r'))
CFG = yaml.safe_load(open('./config/config.yaml', 'r'))
TRAIN_CFG = CFG['TRAIN']

color_dict = KITTI_CFG["color_map"]
learning_map = np.unique(list(KITTI_CFG['learning_map'].values()))
nclasses = learning_map.size

print(f'Loaded {nclasses} classes')
dataset = SemData(split='TRAIN')
val_dataset = SemData(split='VALID')

print('Done loading data')

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


# Set seed
manual_seed = 42
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Set Model, Loss, Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
from model.pspnet import PSPNet
model = PSPNet(layers=50, classes=nclasses, criterion=criterion)
modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
modules_new = [model.ppm, model.cls, model.aux]

base_lr = 0.01
params_list = []
for module in modules_ori:
    params_list.append(dict(params=module.parameters(), lr=base_lr))
for module in modules_new:
    params_list.append(dict(params=module.parameters(), lr=base_lr * 10))
index_split = 5
momentum = 0.9
weight_decay = 0.0001
optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

model = torch.nn.DataParallel(model.to(device))

print('Model initialization done')

logger = get_logger()
logger.info("=> creating model ...")
logger.info("Classes: {}".format(nclasses))
# logger.info(model)


# Set data
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, drop_last=True)

# Main Loop
epochs = 100
aux_weight = 0.4
print_freq = 50
print('Start training')
def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device).long()
        output, main_loss, aux_loss = model(input, target)
        
        main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        intersection, union, target = intersectionAndUnionGPU(output, target, nclasses, 0)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(base_lr, current_iter, max_iter)
        for index in range(0, index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device).long()
        output = model(input)

        loss = criterion(output, target)
        
        loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, nclasses, 0)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % print_freq == 0):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(nclasses):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


save_freq = 5
save_path = './saved'
evaluate = True
for epoch in range(0, epochs):
    epoch_log = epoch + 1
    loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
    
    if (epoch_log % save_freq == 0):
        filename = save_path + '/train_epoch_' + str(epoch_log) + '.pth'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        if epoch_log / save_freq > 2:
            deletename = save_path + '/train_epoch_' + str(epoch_log - save_freq * 2) + '.pth'
            os.remove(deletename)

    if evaluate:
        loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
