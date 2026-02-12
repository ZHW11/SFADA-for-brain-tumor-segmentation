import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import Rot90Transform, MirrorTransform, SpatialTransform
from skimage.transform import resize
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from scipy.ndimage import rotate
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

def get_train_generator(trainloader, num_workers):
    transforms = []

    # 空间变换（仅 XY 方向旋转、缩放等，适合 2D）
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    transforms.append(SpatialTransform(
        data_key='data', label_key='label',
        patch_size=(224, 224),
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=angle, angle_y=None, angle_z=None,
        p_rot_per_axis=1,
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
        random_crop=False,
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False
    ))

    # 其他图像增强
    transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True,
                                            p_per_sample=0.2, p_per_channel=0.5))
    transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                     p_per_channel=0.5, order_downsample=0, order_upsample=3,
                                                     p_per_sample=0.25, ignore_axes=None))
    transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # 镜像增强（仅水平和垂直）
    transforms.append(MirrorTransform(axes=(0, 1), data_key='data', label_key='label'))

    # 标签处理与数据转 tensor
    # transforms.append(RemoveLabelTransform(-1, 0, input_key='label', output_key='label'))
    transforms.append(NumpyToTensor(['data', 'label'], 'float'))  # 使用你的键名

    # 构造增强管道
    transforms = Compose(transforms)

    # 构建 batch generator
    batch_generator = MultiThreadedAugmenter(
        data_loader=trainloader,
        transform=transforms,
        num_processes=num_workers,
        pin_memory=True
    )
    return batch_generator
