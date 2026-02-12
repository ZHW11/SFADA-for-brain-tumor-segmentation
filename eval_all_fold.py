import torch
import os, pickle
import numpy as np
import torch.nn as nn
import statistics as stat

from torch.cuda.amp import autocast
# from torch.distributed.pipeline.sync.checkpoint import checkpoint

from core.config import config
from medpy.metric.binary import *
import torch.backends.cudnn as cudnn

from utils.utils import create_logger, setup_seed
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

def main():

    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    from seg.models.my_model_onlyMonai2 import UNet
    model = UNet(4, 4, feat_size=[48, 96, 192, 384])
    model_name = 'resunet_men_source'
    # model = VSSM(4, 4, 4, drop_path_rate=0.1)
    # model.load_state_dict(torch.load('./experiments/best_model59_zz_tmp.pth', map_location=config.DEVICE))
    # checkpoints = torch.load('experiments/check_model59_select0.01_1e-4.pth', map_location=config.DEVICE)
    checkpoints = torch.load('experiments/source_resunet.pth', map_location=config.DEVICE)
    model.load_state_dict(checkpoints['state_dict'])
    model = model.to(config.DEVICE)
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(log_dir, f'test_{model_name}.log')
    # 存储五折的结果
    all_fold_results = []

    for fold in range(1, 6):
        splits_file = f'splits_fold{fold}.pkl'
        test_result = fold_test(model, logger, config, 'test', metrics=[dc, hd95], fold_output_dir='tmp',
                                model_name=model_name, device=config.DEVICE, splits_file=splits_file)
        all_fold_results.append(test_result)

    # TODO 统计五折的指标，取平均并保存到experiments文件夹中，以txt格式保存
    # 计算五折平均结果
    avg_results = {}

    # 初始化平均结果结构
    for metric in all_fold_results[0]:
        avg_results[metric] = {}
        for region in ['WT', 'ET', 'TC']:
            avg_results[metric][region] = {
                'mean': [],
                'std': []
            }

    # 收集所有折的结果
    for fold_result in all_fold_results:
        for metric in fold_result:
            for region in ['WT', 'ET', 'TC']:
                if region in fold_result[metric]:
                    avg_results[metric][region]['mean'].append(fold_result[metric][region]['mean'])
                    avg_results[metric][region]['std'].append(fold_result[metric][region]['std'])

    # 计算平均值和标准差
    final_results = {}
    for metric in avg_results:
        final_results[metric] = {}
        for region in ['WT', 'ET', 'TC']:
            if avg_results[metric][region]['mean']:
                mean_vals = avg_results[metric][region]['mean']
                std_vals = avg_results[metric][region]['std']

                final_mean = stat.mean(mean_vals) if mean_vals else 0.0
                final_std = stat.mean(std_vals) if std_vals else 0.0

                final_results[metric][region] = {
                    'mean': final_mean,
                    'std': final_std
                }

    # 保存到experiments文件夹
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    output_file = os.path.join(experiments_dir, f'five_fold_avg_results_{model_name}.txt')

    with open(output_file, 'w') as f:
        f.write(f"Five-fold Cross Validation Results for {model_name}\n")
        f.write("=" * 60 + "\n\n")

        for metric in final_results:
            f.write(f"Metric: {metric}\n")
            f.write("-" * 40 + "\n")
            for region in ['WT', 'ET', 'TC']:
                if region in final_results[metric]:
                    result = final_results[metric][region]
                    f.write(f"{region}: Mean = {result['mean']:.4f}, Std = {result['std']:.4f}\n")
            f.write("\n")

        # 添加详细的分折结果
        f.write("\nDetailed Results per Fold:\n")
        f.write("=" * 60 + "\n")
        for fold_idx, fold_result in enumerate(all_fold_results, 1):
            f.write(f"\nFold {fold_idx}:\n")
            f.write("-" * 20 + "\n")
            for metric in fold_result:
                f.write(f"{metric}:\n")
                for region in ['WT', 'ET', 'TC']:
                    if region in fold_result[metric]:
                        result = fold_result[metric][region]
                        f.write(f"  {region}: Mean = {result['mean']:.4f}, Std = {result['std']:.4f}\n")

    logger.info(f"Five-fold cross validation results saved to {output_file}")

    # 在日志中输出平均结果
    logger.info("Five-fold Cross Validation Average Results:")
    for metric in final_results:
        logger.info(f"Metric: {metric}")
        for region in ['WT', 'ET', 'TC']:
            if region in final_results[metric]:
                result = final_results[metric][region]
                logger.info(f"  {region}: Mean = {result['mean']:.4f}, Std = {result['std']:.4f}")


if __name__ == '__main__':
    main()
