import torch
from easydict import EasyDict as edict

config = edict()
config.NUM_WORKERS = 1
config.OUTPUT_DIR = 'experiments'
config.SEED = 42

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = False
config.DATASET = edict()

config.DATASET.ROOT = './data/BraTS2023_MEN/processed'
config.DATASET.SPLITS_NAME = "splits.pkl"
config.DATASET.FOLD_PATH = "folds_data_save5_se6_coreset"
config.DATASET.SLICE_DATA_PATH = "slice"
config.DATASET.TTA_NUM = 8

config.TRAIN = edict()
config.TRAIN.LR = 1e-3
config.TRAIN.WEIGHT_DECAY = 3e-5
config.TRAIN.TRUE_BATCH_SIZE = 4
config.TRAIN.PSEUDO_BATCH_SIZE = 12
config.TRAIN.BATCH_SIZE = 12
config.TRAIN.PATCH_SIZE = [224, 224]
config.TRAIN.NUM_BATCHES = 100
config.TRAIN.EPOCH = 401
config.TRAIN.NUM_ITERS = 20000
config.TRAIN.PARALLEL = True
config.TRAIN.DEVICES = [0]
config.DEVICE = torch.device('cuda:0')
config.TRAIN.USE_AMP = True
config.TRAIN.EARLY_STOPPING = True
config.TRAIN.PATIENCE = 10
