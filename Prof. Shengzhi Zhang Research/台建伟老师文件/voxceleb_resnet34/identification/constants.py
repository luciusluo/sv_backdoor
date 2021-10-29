# Signal Processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
SEGMENT_LEN = 3

# Model
VoxCeleb1_Dir = '/data/corpus/VoxCeleb1/'
NUM_CLASSES = 1251
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 16
BATCH_SIZE = 64
LR_INIT = 0.01
LR_LAST = 0.0001
EPOCH_NUM = 30

# Device configuration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2, 3, 4, 5]   # Use 3 GPUs


