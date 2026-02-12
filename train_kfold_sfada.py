import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from core.config import config
import torch.backends.cudnn as cudnn
from core.scheduler import PolyScheduler
from core.function import inference_kfold
from core.loss import *
from dataset.dataloader_sfada_kfold import get_TrueLabelDataLoader, get_PseudoLabelDataLoader
from dataset.augmenter_sfada_kfold import get_train_generator
from dataset.augmenter_sfada_kfold import get_train_generator_dual
from utils.utils import transunet_save_checkpoint, create_logger, setup_seed
from models.model import SFADA_Net
import matplotlib.pyplot as plt
import os
import statistics as stat
from medpy.metric.binary import dc, jc, hd95, sensitivity, precision, specificity
from torchvision.utils import save_image
from torch.nn.functional import softmax
import torch.nn.functional as F 


@torch.no_grad()
def fold_test(model, logger, config, dataset, metrics, fold_output_dir, model_name, device, splits_file):
    model.eval().to(device)
    perfs = {metric.__name__: {'WT': [], 'ET': [], 'TC': []} for metric in metrics}
    nonline = nn.Softmax(dim=1)

    with open(os.path.join(config.DATASET.ROOT, config.DATASET.FOLD_PATH, splits_file), 'rb') as f:
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
            # Use AMP for forward inference
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

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))


def train_fold(fold, splits_file, device, logger, model_name, fold_selected_path, fold_unselected_path):
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    fold_output_dir = os.path.join(config.OUTPUT_DIR, f'fold{fold}')
    os.makedirs(fold_output_dir, exist_ok=True)

    # Build model
    myModel = SFADA_Net(4, 4, norm_name='instance', feat_size=[48, 96, 192, 384], return_feature=True)
    checkpoints = torch.load('experiments/source_resunet.pth', map_location=device)
    myModel.load_state_dict(checkpoints['state_dict'])
    myModel = myModel.to(device)

    optim_seg = optim.SGD(
        [{'params': myModel.parameters()}],
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        momentum=0.95,
        nesterov=True
    )
    sched_seg = PolyScheduler(optim_seg, t_total=config.TRAIN.NUM_ITERS)
    criterion = DiceCELoss()
    p_criterion = CELoss_p()
    p_trainloader = get_PseudoLabelDataLoader(config, fold_unselected_path=fold_unselected_path)
    t_trainloader = get_TrueLabelDataLoader(config, fold_selected_path=fold_selected_path)
    true_train_generator = get_train_generator(t_trainloader)
    p_train_generator = get_train_generator_dual(p_trainloader)
    best_perf = 0.0
    early_stop_counter = 0
    perf_history = []
    iters = []
    num_iters = config.TRAIN.NUM_ITERS
    val_interval = config.TRAIN.VAL_INTERVAL if hasattr(config.TRAIN, 'VAL_INTERVAL') else 100
    true_lsegs = AverageMeter()
    pseudo_lsegs = AverageMeter()
    contrastive_losses = AverageMeter()
    mask_ratios = AverageMeter()
    scaler = GradScaler(enabled=config.TRAIN.USE_AMP)
    true_loss_weight = getattr(config.TRAIN, 'TRUE_LOSS_WEIGHT', 2)
    pseudo_loss_weight_init = getattr(config.TRAIN, 'PSEUDO_LOSS_WEIGHT', 0.5)
    pseudo_loss_weight_max = getattr(config.TRAIN, 'PSEUDO_LOSS_WEIGHT_MAX', 1)
    warmup_iters = getattr(config.TRAIN, 'PSEUDO_LOSS_WARMUP_ITERS', 400) * config.TRAIN.NUM_BATCHES
    conf_thresh = getattr(config.TRAIN, 'CONF_THRESH', 0.95)
    true_batch_size = config.TRAIN.TRUE_BATCH_SIZE
    pseudo_batch_size = config.TRAIN.PSEUDO_BATCH_SIZE
    print_freq = 25
    current_iter = 0
    model_ema_initialized = False

    while current_iter < num_iters:
        # Calculate pseudo label loss weight
        if current_iter < warmup_iters:
            pseudo_loss_weight = pseudo_loss_weight_init + \
                                 (pseudo_loss_weight_max - pseudo_loss_weight_init) * (current_iter / warmup_iters)
        else:
            pseudo_loss_weight = pseudo_loss_weight_max

        # Get data
        true_data_dict = next(true_train_generator)
        pseudo_data_dict = next(p_train_generator)
        true_image = true_data_dict['data'].to(device)
        true_label = true_data_dict['label'].to(device)
        pseudo_image_strong = pseudo_data_dict['data1'].to(device)
        pseudo_image_weak = pseudo_data_dict['data2'].to(device)
        use_cutmix = getattr(config.TRAIN, 'USE_CUTMIX', True)
        if use_cutmix:
            batch_size = pseudo_image_strong.shape[0]
            cutmix_box = torch.zeros(batch_size, 224, 224, dtype=torch.bool, device=device)
            for b in range(batch_size):
                lam = torch.rand(1).item() * 0.5 + 0.5
                h, w = pseudo_image_strong.shape[2], pseudo_image_strong.shape[3]
                cx, cy = int(torch.rand(1).item() * w), int(torch.rand(1).item() * h)
                bbx1, bby1 = max(0, cx - int(w * lam / 2)), max(0, cy - int(h * lam / 2))
                bbx2, bby2 = min(w, cx + int(w * lam / 2)), min(h, cy + int(h * lam / 2))
                cutmix_box[b, bby1:bby2, bbx1:bbx2] = 1
            pseudo_image_strong_cutmix = pseudo_image_strong.clone()
            pseudo_image_strong_cutmix[cutmix_box.unsqueeze(1).expand_as(pseudo_image_strong) == 1] = \
                pseudo_image_strong.flip(0)[cutmix_box.unsqueeze(1).expand_as(pseudo_image_strong) == 1]
        else:
            pseudo_image_strong_cutmix = pseudo_image_strong

        myModel.train()
        optim_seg.zero_grad()

        with autocast(enabled=config.TRAIN.USE_AMP):
            if current_iter <= 200:
                true_out = myModel(true_image)
                if isinstance(true_out, tuple):
                    true_out, true_feat = true_out  # Assume model returns segmentation output and feature map
                else:
                    true_feat = true_out  # If no separate feature output, use segmentation output
                true_lseg = criterion(true_out, true_label)
                pseudo_lseg = 0.0
                contrastive_loss = 0.0
                mask_ratio = 0.0
                loss = true_loss_weight * true_lseg
            else:
                if not model_ema_initialized:
                    model_ema = deepcopy(myModel).to(device)
                    for param in model_ema.parameters():
                        param.detach_()
                    model_ema_initialized = True
                model_ema.train()
                pred_u_w = model_ema(pseudo_image_weak)
                if isinstance(pred_u_w, tuple):
                    pred_u_w, ema_feat_cutmix = pred_u_w
                pred_u_w = pred_u_w.detach()
                mask_u_w = torch.argmax(pred_u_w, dim=1)
                conf_u_w = F.softmax(pred_u_w, dim=1).max(dim=1)[0]
                ignore_background_conf = False
                if ignore_background_conf:
                    conf_u_w[mask_u_w == 0] = 0
                if use_cutmix:
                    mask_u_w_cutmix = mask_u_w.clone()
                    conf_u_w_cutmix = conf_u_w.clone()
                    mask_u_w_cutmix[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
                    conf_u_w_cutmix[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
                else:
                    mask_u_w_cutmix = mask_u_w
                    conf_u_w_cutmix = conf_u_w

                true_out = myModel(true_image)
                pseudo_out = myModel(pseudo_image_strong_cutmix)
                if isinstance(true_out, tuple):
                    true_out, true_feat = true_out
                else:
                    true_feat = true_out
                if isinstance(pseudo_out, tuple):
                    pseudo_out, pseudo_feat = pseudo_out
                else:
                    pseudo_feat = pseudo_out

                true_lseg = criterion(true_out, true_label)
                pseudo_lseg = p_criterion(pseudo_out, mask_u_w_cutmix)
                # pseudo_lseg = p_criterion(pseudo_out, F.softmax(pred_u_w, dim=1), conf_u_w_cutmix)
                valid_mask = (conf_u_w_cutmix >= conf_thresh)
                pseudo_lseg = pseudo_lseg * valid_mask
                valid_pixels = valid_mask.sum().item()
                pseudo_lseg = pseudo_lseg.sum() / (valid_pixels + 1e-6)
                mask_ratio = valid_pixels / (pseudo_image_strong.shape[2] * pseudo_image_strong.shape[3] * pseudo_batch_size)

                loss = true_loss_weight * true_lseg + pseudo_loss_weight * pseudo_lseg

        scaler.scale(loss).backward()
        scaler.unscale_(optim_seg)
        scaler.step(optim_seg)
        scaler.update()
        if current_iter > 200:
            update_ema_variables(myModel, model_ema, 0.996, current_iter)

        true_lsegs.update(true_lseg.item(), true_batch_size)
        pseudo_lsegs.update(pseudo_lseg.item() if current_iter > 200 else 0.0, pseudo_batch_size)
        mask_ratios.update(mask_ratio if current_iter > 0 else 0.0, pseudo_batch_size)

        if current_iter % print_freq == 0:
            msg = 'Iteration: [{0}/{1}]\t' \
                  'true_lseg {true_lseg.val:.3f} ({true_lseg.avg:.3f})\t' \
                  'pseudo_lseg {pseudo_lseg.val:.6f} ({pseudo_lseg.avg:.6f})\t' \
                  'learning rate: {lr:.6f} \t' \
                  'mask ratio:{mask_ratio:.6f} \t' \
                .format(
                current_iter, num_iters, true_lseg=true_lsegs, pseudo_lseg=pseudo_lsegs,
                 lr=optim_seg.param_groups[0]["lr"], mask_ratio=mask_ratio)
            logger.info(msg)

            bs = true_image.shape[0]
            image_vis = torch.cat(torch.split(true_image, 1, 1))
            label_vis = torch.cat(torch.split(true_label, 1, 1))
            out = torch.argmax(F.softmax(true_out, 1), dim=1, keepdim=True)
            out = torch.cat(torch.split(out, 1, 1))
            save_image(torch.cat([image_vis, label_vis, out], dim=0).data.cpu(),
                       f'tmp/train_{model_name}.png', nrow=bs, scale_each=True, normalize=True)

        sched_seg.step()
        current_iter += 1

        # Validation
        if current_iter % val_interval == 0 or current_iter == num_iters:
            perf = inference_kfold(myModel, logger, config, dataset='test', splits_file=splits_file)

            perf_history.append(perf)
            iters.append(current_iter)
            logger.info(f'Fold {fold}, Iteration {current_iter}, Validation Perf: {perf:.4f}')

            if perf > best_perf:
                best_perf = perf
                early_stop_counter = 0
                transunet_save_checkpoint({
                    'iteration': current_iter,
                    'state_dict': myModel.state_dict(),
                    'optimizer': optim_seg.state_dict(),
                    'scheduler': sched_seg.state_dict(),
                    'perf': perf
                }, True, fold_output_dir, filename=f'check_{model_name}.pth', best_name=f'best_{model_name}.pth')


            else:
                early_stop_counter += 1
                logger.info(f'No improvement in perf for {early_stop_counter} iteration(s).')

            if config.TRAIN.EARLY_STOPPING and early_stop_counter >= config.TRAIN.PATIENCE:
                logger.info(f'Early stopping triggered after {early_stop_counter} iterations without improvement.')
                break

    # Load the best model for testing
    myModel.load_state_dict(torch.load(os.path.join(fold_output_dir, f'best_{model_name}.pth'), map_location=device))

    # Test on test set
    logger.info(f'Fold {fold}, Starting test set evaluation')
    test_results = fold_test(myModel, logger, config, dataset='test',
                             metrics=[dc, jc, hd95, sensitivity, precision, specificity],
                             fold_output_dir=fold_output_dir, model_name=model_name, device=device,
                             splits_file=splits_file)

    logger.info(f'Fold {fold}, Test completed')

    return best_perf, test_results


def main():
    device = config.DEVICE
    fold_perfs = []
    test_results_all = []
    model_name = f"net_{config.DATASET.FOLD_PATH}"
    # model_name = f"tmp"
    log_dir = 'log_fold'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(log_dir, f'train_{model_name}.log')
    print(f"Fold path is {config.DATASET.FOLD_PATH}")

    for fold in range(1, 6):
        fold_path = os.path.join(str(config.DATASET.ROOT), str(config.DATASET.FOLD_PATH))
        train_splits_file = f'fold_{fold}_slice.pkl'
        fold_selected_path = os.path.join(fold_path, f'fold_{fold}_selected_slices.txt')
        fold_unselected_path = os.path.join(fold_path, f'fold_{fold}_unselected_slices.txt')
        splits_file = f'fold_{fold}_volume.pkl'
        logger.info(f'Starting fold {fold} training')
        fold_best_perf, fold_test_results = train_fold(fold, splits_file, device, logger, model_name, fold_selected_path, fold_unselected_path)
        fold_perfs.append(fold_best_perf)
        test_results_all.append(fold_test_results)

        logger.info(f'Fold {fold} completed. Best validation performance: {fold_best_perf:.4f}')

    # Output cross-validation results
    logger.info('5-fold cross-validation completed.')
    with open(os.path.join(config.OUTPUT_DIR, f'{model_name}_crossval_results.txt'), 'w') as f:
        f.write('5-fold cross-validation completed.\n')
        for fold, perf in enumerate(fold_perfs, 1):
            f.write(f'Fold {fold} best validation performance: {perf:.4f}\n')
            logger.info(f'Fold {fold} best validation performance: {perf:.4f}')
        mean_perf = stat.mean(fold_perfs) if fold_perfs else 0.0
        std_perf = stat.stdev(fold_perfs) if len(fold_perfs) > 1 else 0.0
        f.write(f'Mean validation performance across all folds: {mean_perf:.4f} ± {std_perf:.4f}\n')
        logger.info(f'Mean validation performance across all folds: {mean_perf:.4f} ± {std_perf:.4f}')

        # Aggregate test results
        metrics = ['dc', 'jc', 'hd95', 'sensitivity', 'precision', 'specificity']
        aggregated_results = {metric: {'WT': [], 'ET': [], 'TC': []} for metric in metrics}
        for fold_result in test_results_all:
            for metric in metrics:
                for region in ['WT', 'ET', 'TC']:
                    aggregated_results[metric][region].append(fold_result[metric][region]['mean'])

        f.write('Aggregated test results:\n')
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