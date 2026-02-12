import os
import numpy as np
from pathlib import Path
import h5py
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

class TrueLabelDataLoader(SlimDataLoaderBase):
    def __init__(self, data_dir="./seg/data/BraTS2023_SSA/processed/slice_TaP",
                 fold_file="./data/BraTS2023_SSA/folds/fold_1_selected_slices.txt",
                 batch_size=12):
        super(TrueLabelDataLoader, self).__init__(data=None, batch_size=batch_size,
                                                  number_of_threads_in_multithreaded=1)
        self.data_dir = Path(data_dir)

        # 检查 data_dir 是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        # 读取 fold_file 中的切片名称，去除 .h5 扩展名
        try:
            with open(fold_file, 'r') as f:
                selected_slices = [line.strip().replace('.h5', '') for line in f if line.strip()]
            if not selected_slices:
                raise ValueError(f"Fold file {fold_file} is empty or contains only empty lines")
        except FileNotFoundError:
            raise FileNotFoundError(f"Fold file {fold_file} does not exist")

        # 获取所有 .h5 或 .hdf5 文件
        h5_files = list(self.data_dir.glob("*.h5")) + list(self.data_dir.glob("*.hdf5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 or .hdf5 files found in {self.data_dir}")

        # 筛选匹配的 h5 文件
        self.file_list = sorted([f for f in h5_files if f.stem in selected_slices])
        if not self.file_list:
            raise ValueError(f"No .h5 files in {self.data_dir} match the slices in {fold_file}")

        self.indices = list(range(len(self.file_list)))

    def generate_train_batch(self):
        batch_data = []
        batch_labels = []

        # 随机选择批次索引
        selected_indices = np.random.choice(self.indices,
                                            size=min(len(self.indices), self.batch_size),
                                            replace=False)

        for idx in selected_indices:
            file_path = self.file_list[idx]
            with h5py.File(file_path, 'r') as f:
                image = f['slice'][:]  # Shape: (1, 4, H, W)
                label = f['label'][:]  # Shape: (H, W)
                image = image.squeeze(0)  # Shape: (4, H, W)
                label = np.expand_dims(label, axis=0)  # Shape: (1, H, W)
                batch_data.append(image)
                batch_labels.append(label)

        batch_data = np.stack(batch_data, axis=0)  # Shape: (batch_size, 4, H, W)
        batch_labels = np.stack(batch_labels, axis=0)  # Shape: (batch_size, 1, H, W)
        return {
            'data': batch_data,
            'label': batch_labels
        }
class PseudoLabelDataLoader(SlimDataLoaderBase):
    def __init__(self, data_dir="./seg/data/BraTS2023_SSA/processed/slice_TaP",
                 fold_file="./data/BraTS2023_SSA/folds/fold_1_selected_slices.txt",
                 batch_size=12):
        super(PseudoLabelDataLoader, self).__init__(data=None, batch_size=batch_size,
                                                  number_of_threads_in_multithreaded=1)
        self.data_dir = Path(data_dir)

        # 检查 data_dir 是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        # 读取 fold_file 中的切片名称，去除 .h5 扩展名
        try:
            with open(fold_file, 'r') as f:
                selected_slices = [line.strip().replace('.h5', '') for line in f if line.strip()]
            if not selected_slices:
                raise ValueError(f"Fold file {fold_file} is empty or contains only empty lines")
        except FileNotFoundError:
            raise FileNotFoundError(f"Fold file {fold_file} does not exist")

        # 获取所有 .h5 或 .hdf5 文件
        h5_files = list(self.data_dir.glob("*.h5")) + list(self.data_dir.glob("*.hdf5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 or .hdf5 files found in {self.data_dir}")

        # 筛选匹配的 h5 文件
        self.file_list = sorted([f for f in h5_files if f.stem in selected_slices])
        if not self.file_list:
            raise ValueError(f"No .h5 files in {self.data_dir} match the slices in {fold_file}")

        self.indices = list(range(len(self.file_list)))

    def generate_train_batch(self):
        batch_data = []
        batch_labels = []

        # 随机选择批次索引
        selected_indices = np.random.choice(self.indices,
                                            size=min(len(self.indices), self.batch_size),
                                            replace=False)

        for idx in selected_indices:
            file_path = self.file_list[idx]
            with h5py.File(file_path, 'r') as f:
                image = f['slice'][:]  # Shape: (1, 4, H, W)
                label = f['pse_label'][:]  # Shape: (H, W)
                image = image.squeeze(0)  # Shape: (4, H, W)
                label = np.expand_dims(label, axis=0)  # Shape: (1, H, W)
                batch_data.append(image)
                batch_labels.append(label)

        batch_data = np.stack(batch_data, axis=0)  # Shape: (batch_size, 4, H, W)
        batch_labels = np.stack(batch_labels, axis=0)  # Shape: (batch_size, 1, H, W)
        return {
            'data': batch_data,
            'label': batch_labels
        }


def get_TrueLabelDataLoader(config, fold_selected_path):
    slice_path = str(os.path.join(config.DATASET.ROOT, config.DATASET.SLICE_DATA_PATH))
    return TrueLabelDataLoader(slice_path, fold_selected_path, config.TRAIN.TRUE_BATCH_SIZE)
def get_PseudoLabelDataLoader(config, fold_unselected_path):
    slice_path = str(os.path.join(config.DATASET.ROOT, config.DATASET.SLICE_DATA_PATH))
    return PseudoLabelDataLoader(slice_path, fold_unselected_path, config.TRAIN.PSEUDO_BATCH_SIZE)