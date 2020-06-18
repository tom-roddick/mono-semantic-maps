import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST

from .utils import IMAGE_WIDTH, IMAGE_HEIGHT, ARGOVERSE_CLASS_NAMES
from ..utils import decode_binary_labels


class ArgoverseMapDataset(Dataset):

    def __init__(self, argo_loaders, label_root, image_size=[960, 600], 
                 log_names=None):

        self.label_root = label_root
        self.image_size = image_size

        self.examples = dict()
        self.calibs = dict()

        # Preload training examples from Argoverse train and test sets
        self.loaders = argo_loaders
        for split, loader in loaders.items():
            self.preload(split, loader, log_names)


    def preload(self, split, loader, log_names=None):

        # Iterate over sequences
        for log in loader:

            # Check if the log is within the current dataset split
            logid = log.current_log
            if log_names is not None and logid not in log_names:
                continue

            self.calibs[logid] = dict()
            for camera, timestamps in log.image_timestamp_list_sync.items():

                if camera not in RING_CAMERA_LIST:
                    continue

                # Load image paths
                for timestamp in timestamps:
                    self.examples[timestamp] = (split, logid, camera)
    

    def __len__(self):
        return len(self.examples)
    

    def __getitem__(self, timestamp):

        # Get the split, log and camera ids corresponding to the given timestamp
        split, log, camera = self.examples[timestamp]

        image = self.load_image(split, log, camera, timestamp)
        calib = self.load_calib(split, log, camera)
        labels, mask = self.load_labels(split, log, camera, timestamp)

        return image, calib, labels, mask


    def load_image(self, split, log, camera, timestamp):
        
        # Load image
        loader = self.loaders[split]
        image = loader.get_image_at_timestamp(timestamp, camera, log)
        
        # Resize to the desired dimensions
        image = image.resize(self.image_size)

        return to_tensor(image)
    

    def load_calib(self, split, log, camera):

        # Get the loader for the current split
        loader = self.loaders[split]

        # Get intrinsics matrix and rescale to account for downsampling
        calib = loader.get_calibration(camera, log).K[:,:3]
        calib[0] *= self.image_size[0] / IMAGE_WIDTH
        calib[1] *= self.image_size[1] / IMAGE_HEIGHT
        
        # Convert to a torch tensor
        return torch.from_numpy(calib)
    

    def load_labels(self, split, log, camera, timestamp):

        # Construct label path from example data
        label_path = os.path.join(self.label_root, split, log, camera, 
                                  timestamp, f'{camera}_{timestamp}.png')
        
        # Load encoded label image as a torch tensor
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(ARGOVERSE_CLASS_NAMES)
        labels = decode_binary_labels(encoded_labels, num_class+ 1)
        labels, mask = labels[:-1], ~labels[-1]

        return labels, mask
        
    





    