import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

DATASET = "PLAYING_CARDS"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 2
IMAGE_SIZE = 416
NUM_CLASSES = 52
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0  # 1e-4
CONF_THRESHOLD = 0.4
SCALES = [
    IMAGE_SIZE // 32,
    IMAGE_SIZE // 16,
    IMAGE_SIZE // 8,
]  # [13, 26, 52] if future we want to test different scales ex. [20, 40, 80]
PIN_MEMORY = True

COCO_ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
]

# Rescale COCO_ANCHORS to be between [0, 1]
anchors = np.array(COCO_ANCHORS, dtype=float) / IMAGE_SIZE
ANCHORS = np.around(anchors, decimals=2).tolist()
# print(ANCHORS)

CARD_CLASSES = [
    "10c",
    "10d",
    "10h",
    "10s",
    "2c",
    "2d",
    "2h",
    "2s",
    "3c",
    "3d",
    "3h",
    "3s",
    "4c",
    "4d",
    "4h",
    "4s",
    "5c",
    "5d",
    "5h",
    "5s",
    "6c",
    "6d",
    "6h",
    "6s",
    "7c",
    "7d",
    "7h",
    "7s",
    "8c",
    "8d",
    "8h",
    "8s",
    "9c",
    "9d",
    "9h",
    "9s",
    "Ac",
    "Ad",
    "Ah",
    "As",
    "Jc",
    "Jd",
    "Jh",
    "Js",
    "Kc",
    "Kd",
    "Kh",
    "Ks",
    "Qc",
    "Qd",
    "Qh",
    "Qs",
]
