import random
import os, pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import sklearn.model_selection
from scipy.ndimage import binary_fill_holes


def get_bbox(inp):
    coords = np.where(inp != 0)
    minz = np.min(coords[0])
    maxz = np.max(coords[0]) + 1
    minx = np.min(coords[1])
    maxx = np.max(coords[1]) + 1
    miny = np.min(coords[2])
    maxy = np.max(coords[2]) + 1
    return slice(minz, maxz), slice(minx, maxx), slice(miny, maxy)


def convert_seg(seg):
    """ convert brats labels from {0, 1, 2, 3} to {0, 1, 2, 3} """
    new_seg = np.zeros_like(seg)
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 1
    new_seg[seg == 1] = 2
    return new_seg


import hashlib


def remove_duplicate_slices(image, label):
    """
    去除重复的切片，输入:
    - image: ndarray, shape (4, B, H, W)
    - label: ndarray, shape (B, H, W)

    返回:
    - unique_image: shape (4, B_unique, H, W)
    - unique_label: shape (B_unique, H, W)
    """
    seen = set()
    unique_images = []
    unique_labels = []

    B = image.shape[1]  # 切片数
    for i in range(B):
        image_slice = image[:, i]  # shape (4, H, W)
        label_slice = label[i]  # shape (H, W)
        hash_obj = hashlib.md5()
        hash_obj.update(image_slice.tobytes())
        hash_obj.update(label_slice.tobytes())
        hash_key = hash_obj.hexdigest()
        if hash_key not in seen:
            seen.add(hash_key)
            unique_images.append(image_slice)
            unique_labels.append(label_slice)

    if len(unique_images) == 0:
        return None, None

    unique_images = np.stack(unique_images, axis=1)  # shape: (4, B_unique, H, W)
    unique_labels = np.stack(unique_labels, axis=0)  # shape: (B_unique, H, W)
    return unique_images, unique_labels


data_path = '/mnt/d/code/data/BraTS2023MEN/BraTS-MEN-Train'
out_path = './data/BraTS2023_MEN/processed'
os.makedirs(out_path, exist_ok=True)
names = os.listdir(data_path)
assert isinstance(names, list)
random_seed = 42
# 五折交叉验证
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
sorted_names = sorted(names)

# 生成五折分割
fold_splits = []
for fold, (train_idx, test_idx) in enumerate(kf.split(sorted_names)):
    train_names = [sorted_names[i] for i in train_idx]
    test_names = [sorted_names[i] for i in test_idx]
    split_dict = {
        "train": sorted(train_names),
        "test": sorted(test_names)
    }
    fold_splits.append(split_dict)
    # 保存每折的分割结果
    split_path = os.path.join(out_path, f"splits_fold{fold + 1}.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(split_dict, f)
    print(f"Fold {fold + 1} split saved to {split_path}")

# 后续数据处理保持不变
names = os.listdir(data_path)
for name in names:
    flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}-t2f.nii.gz')))
    t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}-t1n.nii.gz')))
    t1ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}-t1c.nii.gz')))
    t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}-t2w.nii.gz')))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}-seg.nii.gz')))
    img = np.stack([flair, t1, t1ce, t2]).astype(np.float32)
    seg = convert_seg(seg)
    # crop foreground regions
    mask = np.zeros_like(seg).astype(bool)
    for i in range(len(img)):
        mask = mask | (img[i] != 0)
    mask = binary_fill_holes(mask)
    bbox = get_bbox(mask)
    img = img[:, bbox[0], bbox[1], bbox[2]]
    seg = seg[bbox[0], bbox[1], bbox[2]]
    mask = mask[bbox[0], bbox[1], bbox[2]]
    # normalization
    for i in range(len(img)):
        img[i][mask] = (img[i][mask] - img[i][mask].min()) / (img[i][mask].max() - img[i][mask].min())
        img[i][mask == 0] = 0
    # compensate label imbalance
    approx_nsamp = 10000
    samp_locs = OrderedDict()
    for cls in [1, 2, 3]:
        locs = np.argwhere(seg == cls)
        nsamp = min(approx_nsamp, len(locs))
        nsamp = max(nsamp, int(np.ceil(0.1 * len(locs))))
        samp = locs[random.sample(range(len(locs)), nsamp)]
        if len(samp) != 0:
            samp_locs[cls] = samp
    data = np.concatenate([img, seg[None]])
    np.save(os.path.join(out_path, f'{name}.npy'), data)
    with open(os.path.join(out_path, f'{name}.pkl'), 'wb') as f:
        pickle.dump(samp_locs, f)
    print(f'{name} is ok!!')