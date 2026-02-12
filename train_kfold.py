import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch.cuda.amp import autocast

from core.config import config
import torch.backends.cudnn as cudnn
from core.scheduler import PolyScheduler
from core.function import inference_FT, train_amp
from core.loss import DiceCELoss
from dataset.dataloader_da import get_trainloader
from dataset.augmenter3 import get_train_generator
from utils.utils import transunet_save_checkpoint, create_logger, setup_seed
from models.model import SFADA_Net
import matplotlib.pyplot as plt
import os
import statistics as stat
from medpy.metric.binary import dc, jc, hd95, sensitivity, precision, specificity


@torch.no_grad()
def fold_test(model, logger, config, dataset, metrics, fold_output_dir, model_name, device, splits_file):
    model.eval().to(device)
    perfs = {metric.__name__: {'WT': [], 'ET': [], 'TC': []} for metric in metrics}
    nonline = nn.Softmax(dim=1)

    with open(os.path.join(config.DATASET.ROOT, splits_file), 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]
    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name + '.npy'))
        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))

        images = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).to(device).float()
        label = data[-1]
        batch_size = 8
        num_samples = images.shape[0]
        out_list = []
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i + batch_size]
            with autocast(enabled=config.TRAIN.USE_AMP):
                batch_out = model(batch_images)
            if isinstance(batch_out, tuple):
                batch_out, _ = batch_out
            out_list.append(batch_out)
        outputs = torch.cat(out_list, dim=0)
        out = nonline(outputs)
        pred = torch.argmax(out, dim=1).cpu().numpy()

        mask = images.permute(1, 0, 2, 3)[0].cpu().numpy() != 0

        for metric in metrics:
            predi = pred if metric.__name__ == 'hd95' else pred[mask]
            labeli = label if metric.__name__ == 'hd95' else label[mask]
            if np.all(predi == 0):
                perfs[metric.__name__]['WT'].append(0.0)
            else:
                perfs[metric.__name__]['WT'].append(metric(predi > 0, labeli > 0))

            if 3 in label:
                y_pred_class3 = (predi == 3).astype(int)
                y_true_class3 = (labeli == 3).astype(int)
                if np.all(y_pred_class3 == 0):
                    perfs[metric.__name__]['ET'].append(100.0 if metric.__name__ == 'hd95' else 0.0)
                else:
                    perfs[metric.__name__]['ET'].append(metric(y_pred_class3, y_true_class3))

            if 2 in label:
                y_pred_class2 = (predi >= 2).astype(int)
                y_true_class2 = (labeli >= 2).astype(int)
                if np.all(y_pred_class2 == 0):
                    perfs[metric.__name__]['TC'].append(100.0 if metric.__name__ == 'hd95' else 0.0)
                else:
                    perfs[metric.__name__]['TC'].append(metric(y_pred_class2, y_true_class2))

        wt_val = perfs['dc']['WT'][-1]
        et_val = perfs['dc']['ET'][-1] if perfs['dc']['ET'] else 'N/A'
        tc_val = perfs['dc']['TC'][-1] if perfs['dc']['TC'] else 'N/A'
        logger.info(f"{name} Dice: WT={wt_val:.4f}, ET={et_val}, TC={tc_val}")

    # Log test results and save to txt
    test_results = {}
    with open(os.path.join(fold_output_dir, f'test_results_{model_name}.txt'), 'w') as f:
        for metric in perfs:
            for region in ['WT', 'ET', 'TC']:
                values = perfs[metric][region]
                mean_val = stat.mean(values) if values else 0.0
                std_val = stat.stdev(values) if len(values) > 1 else 0.0
                f.write(f'{metric} {region} mean / std: {mean_val:.4f} / {std_val:.4f}\n')
                logger.info(f'{metric} {region} mean / std: {mean_val:.4f} / {std_val:.4f}')
                test_results.setdefault(metric, {})[region] = {'mean': mean_val, 'std': std_val}
    return test_results


def train_fold(fold, splits_file, device, logger, model_name):
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    fold_output_dir = os.path.join(config.OUTPUT_DIR, f'fold{fold}')
    os.makedirs(fold_output_dir, exist_ok=True)

    # 构建模型
    myModel = SFADA_Net(4, 4, norm_name='instance', feat_size=[48, 96, 192, 384])
    checkpoints = torch.load('experiments/source.pth', map_location=device)
    myModel.load_state_dict(checkpoints['state_dict'])
    myModel = myModel.to(device)

    for param in myModel.parameters():
        param.requires_grad = True

    optim_seg = optim.SGD(
        filter(lambda p: p.requires_grad, myModel.parameters()),
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        momentum=0.95,
        nesterov=True
    )
    sched_seg = PolyScheduler(optim_seg, t_total=config.TRAIN.EPOCH)
    criterion = DiceCELoss()

    trainloader = get_trainloader(config, splits_name=splits_file)
    train_generator = get_train_generator(trainloader, num_workers=config.NUM_WORKERS)

    best_perf = 0.0
    early_stop_counter = 0
    perf_history = []
    epochs = []
    # logger.info(f'--------第 {fold} 折 source only 结果--------')
    # source_results = fold_test(myModel, logger, config, dataset='test',
    #                          metrics=[dc, jc, hd95, sensitivity, precision, specificity],
    #                          fold_output_dir=fold_output_dir, model_name=model_name, device=device,
    #                          splits_file=splits_file)
    logger.info(f'--------开始第 {fold} 折训练--------')
    for epoch in range(config.TRAIN.EPOCH):
        logger.info(f'Fold {fold}, Epoch {epoch}, learning rate: {optim_seg.param_groups[0]["lr"]:.6f}')

        # 训练
        train_amp(myModel, train_generator, optim_seg, criterion, logger, config, epoch, device,
                  png_name=f'tmp/train_{model_name}.png')
        sched_seg.step()

        if epoch % 1 == 0:
            perf = inference_FT(myModel, logger, config, dataset='test', splits_file=splits_file)
            perf_history.append(perf)
            epochs.append(epoch)
            logger.info(f'Fold {fold}, Epoch {epoch}, Validation Perf: {perf:.4f}')

            if perf > best_perf:
                best_perf = perf
                early_stop_counter = 0
                transunet_save_checkpoint({
                    'epoch': epoch,
                    'state_dict': myModel.state_dict(),
                    'optimizer': optim_seg.state_dict(),
                    'scheduler': sched_seg.state_dict(),
                    'perf': perf
                }, True, fold_output_dir, filename=f'check_{model_name}.pth', best_name=f'best_{model_name}.pth')
            else:
                early_stop_counter += 1
                logger.info(f'No improvement in perf for {early_stop_counter} epoch(s).')

            if config.TRAIN.EARLY_STOPPING and early_stop_counter >= config.TRAIN.PATIENCE:
                logger.info(f'Early stopping triggered after {early_stop_counter} epochs without improvement.')
                break

    # 加载最佳模型进行测试
    best_checkpoint = torch.load(os.path.join(fold_output_dir, f'best_{model_name}.pth'), map_location=device)
    myModel.load_state_dict(best_checkpoint)
    # myModel.load_state_dict(best_checkpoint['state_dict'])

    # 测试集测试
    logger.info(f'Fold {fold}, 开始测试集测试')
    test_results = fold_test(myModel, logger, config, dataset='test',
                             metrics=[dc, jc, hd95, sensitivity, precision, specificity],
                             fold_output_dir=fold_output_dir, model_name=model_name, device=device, splits_file=splits_file)
    logger.info(f'Fold {fold}, 测试完成')

    return best_perf, test_results


def main():
    device = config.DEVICE
    fold_perfs = []
    test_results_all = []
    model_name = f"resunet_FT_fold"
    log_dir = 'log_fold'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(log_dir, f'train_{model_name}.log')

    for fold in range(1, 6):
        # splits_file = f'splits_fold{fold}.pkl'
        splits_file = f'splits_fold{fold}.pkl'


        fold_best_perf, fold_test_results = train_fold(fold, splits_file, device, logger, model_name)
        fold_perfs.append(fold_best_perf)
        test_results_all.append(fold_test_results)

        logger.info(f'第 {fold} 折完成。最佳验证性能: {fold_best_perf:.4f}')

    # 输出交叉验证结果
    logger.info('5折交叉验证完成。')
    with open(os.path.join(config.OUTPUT_DIR, f'{model_name}_crossval_results.txt'), 'w') as f:
        f.write('5折交叉验证完成。\n')
        for fold, perf in enumerate(fold_perfs, 1):
            f.write(f'第 {fold} 折最佳验证性能: {perf:.4f}\n')
            logger.info(f'第 {fold} 折最佳验证性能: {perf:.4f}')
        mean_perf = stat.mean(fold_perfs) if fold_perfs else 0.0
        std_perf = stat.stdev(fold_perfs) if len(fold_perfs) > 1 else 0.0
        f.write(f'所有折验证性能均值: {mean_perf:.4f} ± {std_perf:.4f}\n')
        logger.info(f'所有折验证性能均值: {mean_perf:.4f} ± {std_perf:.4f}')

        # 聚合测试结果
        metrics = ['dc', 'jc', 'hd95', 'sensitivity', 'precision', 'specificity']
        aggregated_results = {metric: {'WT': [], 'ET': [], 'TC': []} for metric in metrics}
        for fold_result in test_results_all:
            for metric in metrics:
                for region in ['WT', 'ET', 'TC']:
                    aggregated_results[metric][region].append(fold_result[metric][region]['mean'])

        f.write('聚合测试结果:\n')
        for metric in metrics:
            f.write(f'------------ {metric} ------------\n')
            for region in ['WT', 'ET', 'TC']:
                means = aggregated_results[metric][region]
                mean_val = stat.mean(means) if means else 0.0
                std_val = stat.stdev(means) if len(means) > 1 else 0.0
                f.write(f'{region} mean / std: {mean_val:.4f} / {std_val:.4f}\n')
                logger.info(f'{metric} {region} mean / std: {mean_val:.4f} / {std_val:.4f}')


if __name__ == '__main__':
    main()