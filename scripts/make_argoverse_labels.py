import os
import sys
import numpy as np
from PIL import Image
from progressbar import ProgressBar

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, \
    encode_binary_labels
from src.data.argoverse.utils import get_object_masks, get_map_mask


def process_split(split, map_data, config):

    # Create an Argoverse loader instance
    path = os.path.join(os.path.expandvars(config.argoverse.root), split)
    print("Loading Argoverse tracking data at " + path)
    loader = ArgoverseTrackingLoader(path)

    for scene in loader:
        process_scene(split, scene, map_data, config)


def process_scene(split, scene, map_data, config):

    print("\n\n==> Processing scene: " + scene.current_log)

    i = 0
    progress = ProgressBar(
        max_value=len(RING_CAMERA_LIST) * scene.num_lidar_frame)
    
    # Iterate over each camera and each frame in the sequence
    for camera in RING_CAMERA_LIST:
        for frame in range(scene.num_lidar_frame):
            progress.update(i)
            process_frame(split, scene, camera, frame, map_data, config)
            i += 1
            

def process_frame(split, scene, camera, frame, map_data, config):

    # Compute object masks
    masks = get_object_masks(scene, camera, frame, config.map_extents,
                             config.map_resolution)
    
    # Compute drivable area mask
    masks[0] = get_map_mask(scene, camera, frame, map_data, config.map_extents,
                            config.map_resolution)
    
    # Ignore regions of the BEV which are outside the image
    calib = scene.get_calibration(camera)
    masks[-1] |= ~get_visible_mask(calib.K, calib.camera_config.img_width,
                                   config.map_extents, config.map_resolution)
    
    # Ignore regions of the BEV which are occluded (based on LiDAR data)
    lidar = scene.get_lidar(frame)
    cam_lidar = calib.project_ego_to_cam(lidar)
    masks[-1] |= get_occlusion_mask(cam_lidar, config.map_extents, 
                                    config.map_resolution)
    
    # Encode masks as an integer bitmask
    labels = encode_binary_labels(masks)

    # Create a filename and directory
    timestamp = str(scene.image_timestamp_list_sync[camera][frame])
    output_path = os.path.join(config.argoverse.label_root, split, 
                               scene.current_log, camera, 
                               f'{camera}_{timestamp}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save encoded label file to disk
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
    

if __name__ == '__main__':

    config = get_default_configuration()
    config.merge_from_file('configs/datasets/argoverse.yml')

    # Create an Argoverse map instance
    map_data = ArgoverseMap()

    for split in ['train', 'val']:
        process_split(split, map_data, config)


