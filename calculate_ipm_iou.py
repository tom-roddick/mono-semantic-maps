import numpy as np
import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import scipy.ndimage

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise
import matplotlib.pyplot as plt

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST

from src.data.argoverse.utils import IMAGE_WIDTH, IMAGE_HEIGHT, ARGOVERSE_CLASS_NAMES
from src.data.utils import decode_binary_labels
import cv2

# def resize_mask(mask):
# # warp the masks
#     image = scipy.ndimage.rotate(cv2.resize(mask, (196, 200)), angle = 270)

#     # color = [0, 0, 0]
#     # new_im = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT,
#     #     value=color)
#     # image = cv2.resize(new_im, (196, 200))
#     # M = np.float32([[1, 0, 0], [0, 1, -35]])
#     # image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

#     M = np.float32([[1,0,0], [0,1,-25]])
#     image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#     image = image[:-25, :]
#     image = cv2.copyMakeBorder(image, 0, 25, 0, 0, cv2.BORDER_REPLICATE)
#     return image

# cityscape labels: ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
# class indices start from 1 in the bev_mask
# bev_mask = np.load("sem_test_warped.npy")
# bev_mask = resize_mask(bev_mask)

# Set up path to data
ARGOVERSE_DATA_ROOT = './data/argo/argoverse-tracking/'
split = 'val'
split_data_dir = ARGOVERSE_DATA_ROOT + split
data_root_split = os.path.join(ARGOVERSE_DATA_ROOT, split)
# log_id = '10b8dee6-778f-33e4-a946-d842d2d9c3d7'
log_id = '5ab2697b-6e3e-3454-a36a-aba2c6f27818'
camera_name = 'ring_front_center'
# cam_timestamp = '315968229010581848'

# img_fpath = f'{split_data_dir}/{log_id}/{camera_name}/{camera_name}_{cam_timestamp}.jpg'
LABEL_ROOT = 'data/argo/labels/'
MASK_ROOT = './data/argo/masks_warped'
mask_folder = os.path.join(MASK_ROOT, split, log_id, camera_name)

def load_labels(split, log, camera, timestamp):
    # Construct label path from example data
                            
    label_path = os.path.join(LABEL_ROOT, split, log, camera, 
                                f'{camera}_{timestamp}.png')
    
    if not os.path.exists(label_path):
        return None, None
    # Load encoded label image as a torch tensor
    encoded_labels = to_tensor(Image.open(label_path)).long()

    # Decode to binary labels
    num_class = len(ARGOVERSE_CLASS_NAMES)
    labels = decode_binary_labels(encoded_labels, num_class+ 1)
    labels, mask = labels[:-1], ~labels[-1]

    return labels, mask

mask_names = [f for f in os.listdir(mask_folder) if f.endswith('.npy')]
print(len(mask_names))
test_path = os.path.join(LABEL_ROOT, split, log_id, camera_name)
print(len(os.listdir(test_path)))

confusion = BinaryConfusionMatrix(8)
for mask_name in tqdm(mask_names):
    cam_timestamp = mask_name.split('.')[0].split('_')[-1]
    mask = np.load(os.path.join(mask_folder, mask_name)).astype(np.uint8)

    drivable_mask = mask == 1
    vehicle_mask = mask == 14
    pedestrain_mask = mask == 12

    preds = np.zeros((8, drivable_mask.shape[0], drivable_mask.shape[1]))
    preds[0] = drivable_mask
    preds[1] = vehicle_mask
    preds[2] = pedestrain_mask

    preds = torch.LongTensor(preds).unsqueeze(0)

    labels, mask = load_labels(split, log_id, camera_name, cam_timestamp)
    if labels == None:
        continue

    labels = labels.unsqueeze(0)
    mask = mask.unsqueeze(0)
    preds = torch.LongTensor(preds).unsqueeze(0)
    confusion.update(preds, labels, mask)


# Display and record epoch IoU scores
class_names = ARGOVERSE_CLASS_NAMES
for name, iou_score in zip(class_names, confusion.iou):
    print('{}: {}'.format(name, iou_score))

# fig = plt.figure(figsize=(15,10))
# # drivable_mask = np.flipud(drivable_mask)
# # drivable_mask = scipy.ndimage.rotate(drivable_mask, angle=90)
# plt.imshow(drivable_mask)
# plt.savefig("drivable_mask.jpg")

# fig = plt.figure(figsize=(15,10))
# # vehicle_mask = np.flipud(vehicle_mask)
# # vehicle_mask = scipy.ndimage.rotate(vehicle_mask, angle=90)
# plt.imshow(vehicle_mask)
# plt.savefig("vehicle_mask.jpg")

# fig = plt.figure(figsize=(15,10))
# # pedestrain_mask = np.flipud(pedestrain_mask)
# # pedestrain_mask = scipy.ndimage.rotate(pedestrain_mask, angle=90)
# plt.imshow(pedestrain_mask)
# plt.savefig("pedestrain_mask.jpg")


# ARGOVERSE_DATA_ROOT = './data/argo/argoverse-tracking/'
# camera_name = 'ring_front_center'
# cam_timestamp = '315968229010581848'
# log_id = '10b8dee6-778f-33e4-a946-d842d2d9c3d7'

# split_data_dir = f'{ARGOVERSE_DATA_ROOT}/train'
# # img_fpath = f'{split_data_dir}/{log_id}/{camera_name}/{camera_name}_{cam_timestamp}.jpg'
# LABEL_ROOT = 'data/argo/labels/'

# preds = np.zeros((8, drivable_mask.shape[0], drivable_mask.shape[1]))
# preds[0] = drivable_mask
# preds[1] = vehicle_mask
# preds[2] = pedestrain_mask


# labels, mask = load_labels('train', log_id, camera_name, cam_timestamp)

# labels = labels.unsqueeze(0)
# mask = mask.unsqueeze(0)
# preds = torch.LongTensor(preds).unsqueeze(0)

# print(labels.shape)
# print(mask.shape)
# confusion = BinaryConfusionMatrix(8)
# confusion.update(preds, labels, mask)

# Display and record epoch IoU scores
# class_names = ARGOVERSE_CLASS_NAMES
# for name, iou_score in zip(class_names, confusion.iou):
#     print('{}: {}'.format(name, iou_score))

# fig = plt.figure(figsize=(15,10))
# plt.imshow(labels[0][0].numpy())
# plt.savefig("drivable_label.jpg")

# fig = plt.figure(figsize=(15,10))
# plt.imshow(labels[0][1].numpy())
# plt.savefig("vehicle_label.jpg")
