from copy import deepcopy
import torch
import os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from medpy.metric.binary import dc
from utils.utils import AverageMeter
from torchvision.utils import save_image
from torch.nn.functional import softmax


@torch.no_grad()
def inference_kfold(model, logger, config, dataset, splits_file=None):
    """
    修改后的推理函数，支持交叉验证

    参数:
        model: 模型
        logger: 日志记录器
        config: 配置对象
        dataset: 数据集类型('train', 'val', 'test'等)
        splits_file: 指定使用的splits文件路径(用于交叉验证)
    """
    model.eval()
    perfs = {'WT': AverageMeter(), 'ET': AverageMeter(), 'TC': AverageMeter()}
    nonline = nn.Softmax(dim=1)

    # 使用传入的splits文件或默认的config中的文件
    if splits_file:
        splits_path = os.path.join(config.DATASET.ROOT, config.DATASET.FOLD_PATH, splits_file)
    else:
        splits_path = os.path.join(config.DATASET.ROOT, config.DATASET.SPLITS_NAME)
    # splits_path = splits_file if splits_file else os.path.join(config.DATASET.ROOT, config.DATASET.SPLITS_NAME)
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]

    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name + '.npy'))
        # pad slice
        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))

        # run inference
        D, H, W = data[0].shape
        batch_size = 16  # 可根据显存大小调整

        # 分批处理并收集预测结果
        pred_list = []
        for i in range(0, D, batch_size):
            end = min(i + batch_size, D)
            image_batch = data[:-1, i:end, :, :]
            image_batch = torch.from_numpy(image_batch).permute(1, 0, 2, 3).to(config.DEVICE).float()
            with autocast(enabled=config.TRAIN.USE_AMP):
                out_batch = model(image_batch)
            if isinstance(out_batch, tuple):
                out_batch, _ = out_batch
            out_batch = nonline(out_batch)
            pred_batch = torch.argmax(out_batch, dim=1).cpu().numpy()
            del image_batch, out_batch
            pred_list.append(pred_batch)
        pred = np.concatenate(pred_list, axis=0)
        label = data[-1]
        # quantitative analysis
        perfs['WT'].update(dc(pred > 0, label > 0))
        if 3 in label:
            perfs['ET'].update(dc(pred == 3, label == 3))
        if 2 in label:
            perfs['TC'].update(dc(pred >= 2, label >= 2))

    for c in perfs.keys():
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')

    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf

@torch.no_grad()
def inference_FT(model, logger, config, dataset, splits_file=None):
    """
    修改后的推理函数，支持交叉验证

    参数:
        model: 模型
        logger: 日志记录器
        config: 配置对象
        dataset: 数据集类型('train', 'val', 'test'等)
        splits_file: 指定使用的splits文件路径(用于交叉验证)
    """
    model.eval()
    perfs = {'WT': AverageMeter(), 'ET': AverageMeter(), 'TC': AverageMeter()}
    nonline = nn.Softmax(dim=1)

    # 使用传入的splits文件或默认的config中的文件
    if splits_file:
        splits_path = os.path.join(config.DATASET.ROOT, splits_file)
    else:
        splits_path = os.path.join(config.DATASET.ROOT, config.DATASET.SPLITS_NAME)
    # splits_path = splits_file if splits_file else os.path.join(config.DATASET.ROOT, config.DATASET.SPLITS_NAME)
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]

    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name + '.npy'))
        # pad slice
        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))

        # run inference
        D, H, W = data[0].shape
        batch_size = 16  # 可根据显存大小调整

        # 分批处理并收集预测结果
        pred_list = []
        for i in range(0, D, batch_size):
            end = min(i + batch_size, D)
            image_batch = data[:-1, i:end, :, :]
            image_batch = torch.from_numpy(image_batch).permute(1, 0, 2, 3).to(config.DEVICE).float()
            with torch.no_grad():
                out_batch = model(image_batch)
            if isinstance(out_batch, tuple):
                out_batch, _ = out_batch
            out_batch = nonline(out_batch)
            pred_batch = torch.argmax(out_batch, dim=1).cpu().numpy()
            del image_batch, out_batch
            pred_list.append(pred_batch)
        pred = np.concatenate(pred_list, axis=0)
        label = data[-1]
        # quantitative analysis
        perfs['WT'].update(dc(pred > 0, label > 0))
        if 3 in label:
            perfs['ET'].update(dc(pred == 3, label == 3))
        if 2 in label:
            perfs['TC'].update(dc(pred >= 2, label >= 2))

    for c in perfs.keys():
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')

    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf



from torch.cuda.amp import autocast, GradScaler

def train_amp(model, train_generator, optimizer, criterion, logger, config, epoch, device, png_name=f'tmp/transunet_all_train.png'):
    unet = model
    unet.train()
    print_freq = 25
    lsegs = AverageMeter()
    num_iter = config.TRAIN.NUM_BATCHES
    scaler = GradScaler(enabled=config.TRAIN.USE_AMP)

    for i in range(num_iter):
        data_dict = next(train_generator)
        image = data_dict['data'].to(device)
        label = data_dict['label'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.TRAIN.USE_AMP):
            outs = unet(image)
            lseg = criterion(outs, label)
            loss = lseg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(unet.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        lsegs.update(lseg.item(), config.TRAIN.BATCH_SIZE)

        if i % print_freq == 0:
            msg = 'epoch: [{0}][{1}/{2}]\t' \
                  'lseg {lseg.val:.3f} ({lseg.avg:.3f})\t'.format(
                      epoch, i, num_iter, lseg=lsegs)
            logger.info(msg)

            bs = image.shape[0]
            image = torch.cat(torch.split(image, 1, 1))
            label = torch.cat(torch.split(label, 1, 1))
            out = torch.argmax(torch.softmax(outs, 1), dim=1, keepdim=True)
            out = torch.cat(torch.split(out, 1, 1))
            save_image(torch.cat([image, label, out], dim=0).data.cpu(), png_name, nrow=bs,
                       scale_each=True, normalize=True)


