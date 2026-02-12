import os
import random
import numpy as np
from kneed import KneeLocator
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn
import pickle
from core.config import config
import h5py

def softmax_entropy(prob, softmax=True):
    """Calculate entropy of a probability distribution."""
    if softmax:
        prob = torch.softmax(prob, dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    return entropy

from skimage.segmentation import slic
from skimage.util import img_as_float


def select_topk_by_kmeans(scores, n_clusters=2, min_k=15):
    """
    用 K-means 把区域平均熵分成 2 类，返回“高不确定度”那一类对应的索引。
    如果高不确定类为空，则回退到取前 10 % 的策略。

    Parameters
    ----------
    scores : 1-D np.ndarray
        每个区域的平均熵值。
    n_clusters : int, default=2
        KMeans 的聚类簇数，通常 2 即可区分高低。

    Returns
    -------
    k : int
        选出的 top-k 区域个数（高不确定簇的大小）。
    sorted_scores : np.ndarray
        降序排列后的 scores（与原 detect_k 接口保持一致）。
    """
    scores = np.asarray(scores)
    sorted_scores = np.sort(scores)[::-1]  # 降序

    # ---- 1. K-means 聚类 ----
    scores_2d = scores.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(scores_2d)

    # ---- 2. 找出“高不确定”簇 ----
    # 簇中心越大 => 越不确定
    cluster_order = np.argsort(km.cluster_centers_.ravel())[::-1]
    high_label = cluster_order[0]  # 最高簇标签
    topk_idx = np.where(labels == high_label)[0]
    k = max(min_k, len(topk_idx))

    return k, sorted_scores

def compute_region_uncertainty_map(entropy_map_np, num_segments=500):
    """
    输入:
        entropy_map_np: 单张熵图 (H, W)，类型为 numpy 数组
    输出:
        uncertainty_score: 最终的不确定性指标
    """
    # 归一化熵图到 [0,1] 以适配 SLIC
    if entropy_map_np.ndim == 3:
        entropy_map_np = entropy_map_np.squeeze(0)  # 变成 (H, W)
    entropy_map_np = img_as_float(entropy_map_np)
    # 使用 SLIC 超像素分割
    segments = slic(entropy_map_np, n_segments=num_segments, compactness=0.05, start_label=0, channel_axis=None)
    region_scores = []
    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        region_entropy = entropy_map_np[mask]
        region_scores.append(region_entropy.mean())
    sorted_scores = np.sort(region_scores)[::-1]
    k, sorted_scores = select_topk_by_kmeans(sorted_scores, n_clusters=2, min_k=25)
    uncertainty_score = sorted_scores[:k].mean()
    return uncertainty_score


random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
from seg.models.my_model_onlyMonai2 import UNet
net = UNet(4, 4, norm_name='instance', feat_size=[48, 96, 192, 384], select_sample=True)
check = torch.load("./experiments/source_resunet.pth", map_location=config.DEVICE)
net.load_state_dict(check['state_dict'])
net = net.to(config.DEVICE)
net.eval()
file_path = config.DATASET.ROOT
npy_files = [f for f in os.listdir(file_path) if f.endswith(".npy")]
# 提取不带扩展名的文件名
file_names = [os.path.splitext(f)[0] for f in npy_files]
file_names.sort()
# output_path = "./data/BraTS2023_SSA/processed/slice_TaP"
output_path = os.path.join(str(config.DATASET.ROOT), str(config.DATASET.SLICE_DATA_PATH))
os.makedirs(output_path, exist_ok=True)
for case in tqdm(file_names):
    data = np.load(os.path.join(config.DATASET.ROOT, f'{case}.npy'))
    shape = np.array(data.shape[2:])  # Shape: (H, W)
    pad_length = [224, 224] - shape
    pad_left = pad_length // 2
    pad_right = pad_length - pad_left
    pad_left = np.clip(pad_left, 0, pad_length)
    pad_right = np.clip(pad_right, 0, pad_length)
    data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))

    image = data[:-1]  # Shape: (4, B, H, W)
    label = data[-1]  # Shape: (B, H, W)
    image = np.transpose(image, (1, 0, 2, 3))  # Shape: (B, 4, H, W)
    image_tensor = torch.from_numpy(image).float().to(config.DEVICE)
    for ind in range(image.shape[0]):
        img_name = f'{case}_slice{ind}'
        slice = image_tensor[ind:ind+1, :, :, :]  # Shape: (1, 4, H, W)
        out_main, features = net(slice)
        feature = features[0].squeeze().detach().cpu().numpy().flatten()
        output_prob = torch.softmax(out_main, dim=1)  # Shape: (1, num_classes, H, W)
        pse_label = torch.argmax(output_prob, dim=1).squeeze().cpu().numpy()
        entropy = softmax_entropy(output_prob, softmax=False)
        entropy_np = entropy.cpu().detach().numpy()
        uncertainty = compute_region_uncertainty_map(entropy_np)
        # 保存为 HDF5 文件
        h5_file_path = os.path.join(output_path, f'{img_name}.h5')
        with h5py.File(h5_file_path, 'w') as f:
            f.create_dataset('slice', data=slice.cpu().numpy()) # Shape: (1, 4, H, W)
            f.create_dataset('label', data=label[ind])   # Shape: (H, W)
            f.create_dataset('pse_label', data=pse_label)  # Shape: (H, W)
            f.create_dataset('feature', data=feature)  # Shape: 150528(1*768*14*14)
            f.create_dataset('uncertainty', data=uncertainty)  # Shape: 1

