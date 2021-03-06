from yacs.config import CfgNode as CN

_C = CN()
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.DEVICE = 'cuda'
_C.DEVICE_ID = '0'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'ResModel'
_C.MODEL.ARCH = 'ResNet50'
_C.MODEL.STRIDE = 2

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
# _C.INPUT.HF_PROB = 0.5

# Values to be used for image normalization
# _C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
# _C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# Value of padding size
# _C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.NAME = ''
# Setup storage directory for dataset
_C.DATASETS.ROOT = '/home/hzh/data/neat'
_C.DATASETS.TRAIN_DIR = 'Train_train'
_C.DATASETS.VAL_DIR = 'Train_val'
_C.DATASETS.TOTAL_TRAIN_DIR = 'Train_Folder'

_C.DATASETS.TEST_DIR = 'Test'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of instance for one batch
_C.DATALOADER.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Sampler for data loading
_C.SOLVER.LOSS = 'softmax'
_C.SOLVER.MAX_EPOCHS = 120
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.BASE_LR = 3e-4

# SGD
# _C.SOLVER.BASE_LR = 0.01
_C.SOLVER.NESTEROV = True
_C.SOLVER.MOMENTUM = 0.9

# Adam
_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.EVAL_PERIOD = 50
_C.SOLVER.PRINT_FREQ = 10

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'StepLR'
_C.SCHEDULER.STEP = 5
_C.SCHEDULER.GAMMA = 0.1

# Warm up factor
_C.SCHEDULER.WARMUP_FACTOR = 100
# iterations of warm up
_C.SCHEDULER.WARMUP_ITERS = 20

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 128
