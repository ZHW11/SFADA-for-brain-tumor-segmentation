import os
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import h5py
import pickle
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from core.config import config


# 配置路径和参数
DATA_ROOT = os.path.join(str(config.DATASET.ROOT), str(config.DATASET.SLICE_DATA_PATH))
OUTPUT_DIR = os.path.join(str(config.DATASET.ROOT), str(config.DATASET.FOLD_PATH))
label_ratio = 0.05 # TODO  !!!!
NUM_FOLDS = 5
ALPHA = 0.5


def read_h5_file(file_path):
    """读取单个H5文件并返回所需数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            feature = f['feature'][:]  # Shape: (150528,)
            uncertainty = f['uncertainty'][()]  # Shape: scalar
            pse_label = f['pse_label'][()]
            return os.path.basename(file_path), uncertainty, feature, pse_label
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
import torch
import torch.nn as nn
def spatial_average_pooling_torch(features, pool_size=2):

    n_samples, channels, height, width = features.shape
    if height % pool_size != 0 or width % pool_size != 0:
        raise ValueError(f"Feature map size ({height}, {width}) must be divisible by pool_size ({pool_size})")
    features_tensor = torch.from_numpy(features).float()  # 形状 (n_samples, channels, height, width)
    avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
    pooled_tensor = avg_pool(features_tensor)  # 形状 (n_samples, channels, height//pool_size, width//pool_size)
    pooled_features = pooled_tensor.numpy().reshape(n_samples, -1)  # 形状 (n_samples, channels * out_height * out_width)

    return pooled_features


from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

def coreset_select_data_by_uncertainty(combined_data, budget, existing_indices=None, alpha=0.5):
    features = np.array([item[2] for item in combined_data])
    uncertainties = np.array([item[1] for item in combined_data])
    uncertainties = (uncertainties - uncertainties.min()) / (
                uncertainties.max() - uncertainties.min() + 1e-8)
    names = [item[0] for item in combined_data]
    n_samples = features.shape[0]

    if budget <= 0:
        return [combined_data[i] for i in existing_indices or []], [names[i] for i in existing_indices or []]

    distances = euclidean_distances(features)

    selected = set(existing_indices or [])
    remaining = set(range(n_samples)) - selected


    if not selected and remaining:
        initial = np.argmax([uncertainties[i] for i in remaining])
        selected.add(initial)
        remaining.remove(initial)
        budget -= 1

    for _ in tqdm(range(budget), desc="Core-set selection"):
        if not remaining:
            break
        rem_list = list(remaining)
        sel_list = list(selected)
        sub_dist = distances[np.ix_(rem_list, sel_list)]
        min_dist = np.min(sub_dist, axis=1)
        min_dist = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-8)
        scores = alpha * min_dist + (1 - alpha) * np.array([uncertainties[i] for i in rem_list])
        new_idx = rem_list[np.argmax(scores)]
        selected.add(new_idx)
        remaining.remove(new_idx)

    selected = sorted(selected)
    return [combined_data[i] for i in selected], [names[i] for i in selected]


def select_data_by_uncertainty(combined_data, budget):
    uncertainties = np.array([data[1] for data in combined_data])
    names = [data[0] for data in combined_data]
    sorted_indices = np.argsort(-uncertainties)
    # 取前 budget 个索引对应的 names
    selected_indices = sorted_indices[:budget]
    selected_data = [combined_data[i] for i in selected_indices]
    selected_names = [names[i] for i in selected_indices]
    return selected_data, selected_names


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有H5文件
    h5_files = [f for f in os.listdir(DATA_ROOT) if f.endswith(".h5")]
    h5_files.sort()

    # 提取案例ID (e.g., BraTS-SSA-00002-000)
    case_ids = sorted(set([f.split('_')[0] for f in h5_files]))

    # 五折划分
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(case_ids)):
        print(f"Processing Fold {fold + 1}/{NUM_FOLDS}")
        train_cases = [case_ids[i] for i in train_idx]
        test_cases = [case_ids[i] for i in test_idx]

        # 获取训练集和测试集对应的H5文件
        train_files = [f for f in h5_files if f.split('_')[0] in train_cases]
        test_files = [f for f in h5_files if f.split('_')[0] in test_cases]

        # 保存划分结果
        fold_data = {'train': train_files, 'test': test_files}
        with open(os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_slice.pkl'), 'wb') as f:
            pickle.dump(fold_data, f)
        fold_data = {'train': train_cases, 'test': test_cases}
        with open(os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_volume.pkl'), 'wb') as f:
            pickle.dump(fold_data, f)

        combined_data = []
        for file_name in tqdm(train_files, desc=f"Reading H5 files for Fold {fold + 1}"):
            result = read_h5_file(os.path.join(DATA_ROOT, file_name))
            if result is not None:
                combined_data.append(result)
        budget = int(len(combined_data) * label_ratio)
        step2_data, selected_names = select_data_by_uncertainty(combined_data, int(budget * 5))
        print(f"step2 selected {len(step2_data)} samples")
        _, selected_names = coreset_select_data_by_uncertainty(step2_data, budget)
        print(f"final selected {len(selected_names)} samples")
        selected_names = sorted(selected_names)
        with open(os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_selected_slices.txt'), 'w') as f:
            for name in selected_names:
                f.write(name + '\n')
        # 保存未选中的切片名称
        all_train_names = [data[0] for data in combined_data]
        unselected_names = set(all_train_names) - set(selected_names)
        unselected_names = sorted(unselected_names)
        with open(os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_unselected_slices.txt'), 'w') as f:
            for name in unselected_names:
                f.write(name + '\n')

if __name__ == "__main__":
    main()