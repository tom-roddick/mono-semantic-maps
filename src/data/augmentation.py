import torch
from torch.utils.data import Dataset
import random

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True):
        self.dataset = dataset
        self.hflip = hflip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, calib, labels, mask, ipm = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image, labels, mask, ipm = random_hflip(image, labels, mask, ipm)

        return image, calib, labels, mask, ipm

    
def random_hflip(image, labels, mask, ipm):
#     coin = random.randint(0,1)
#     if coin:
    image = torch.flip(image, (-1,))
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    ipm = torch.flip(ipm, (-1,))

    return image, labels, mask, ipm