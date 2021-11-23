import numpy as np
from scipy.ndimage import affine_transform
from ..utils import render_polygon


# Define Argoverse-specific constants
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

ARGOVERSE_CLASS_NAMES = [
    'drivable_area', 'vehicle', 'pedestrian', 'large_vehicle', 'bicycle', 'bus',
    'trailer', 'motorcycle',
]

ARGOVERSE_CLASS_MAPPING = {
    'VEHICLE' : 'vehicle',
    'PEDESTRIAN' : 'pedestrian',
    # 'ON_ROAD_OBSTACLE' : 'ignore',
    'LARGE_VEHICLE' : 'large_vehicle',
    'BICYCLE' : 'bicycle',
    'BICYCLIST' : 'bicycle',
    'BUS' : 'bus',
    # 'OTHER_MOVER' : 'ignore',
    'TRAILER' : 'trailer',
    'MOTORCYCLIST' : 'motorcycle',
    'MOPED' : 'motorcycle',
    'MOTORCYCLE' : 'motorcycle',
    # 'STROLLER' : 'ignore',
    'EMERGENCY_VEHICLE' : 'vehicle',
    # 'ANIMAL' : 'ignore',
}

def argoverse_name_to_class_id(name):
    if name in ARGOVERSE_CLASS_MAPPING:
        return ARGOVERSE_CLASS_NAMES.index(ARGOVERSE_CLASS_MAPPING[name])
    else:
        return -1


def get_object_masks(scene, camera, frame, extents, resolution):

    # Get the dimensions of the birds-eye-view mask
    x1, z1, x2, z2 = extents
    mask_width = int((x2 - x1) / resolution)
    mask_height = int((z2 - z1) / resolution)

    # Initialise masks
    num_class = len(ARGOVERSE_CLASS_NAMES)
    masks = np.zeros((num_class + 1, mask_height, mask_width), dtype=np.uint8)

    # Get calibration information
    calib = scene.get_calibration(camera)

    # Iterate over objects in the scene
    for obj in scene.get_label_object(frame):

        # Get the bounding box and convert into camera coordinates
        bbox = obj.as_2d_bbox()[[0, 1, 3, 2]]
        cam_bbox = calib.project_ego_to_cam(bbox)[:, [0, 2]]

        # Render the bounding box to the appropriate mask layer
        class_id = argoverse_name_to_class_id(obj.label_class)
        render_polygon(masks[class_id], cam_bbox, extents, resolution)
    
    return masks.astype(np.bool)


def get_map_mask(scene, camera, frame, map_data, extents, resolution):

    # Get the dimensions of the birds-eye-view mask
    x1, z1, x2, z2 = extents
    mask_width = int((x2 - x1) / resolution)
    mask_height = int((z2 - z1) / resolution)

    # Get rasterised map
    city_mask, map_tfm = map_data.get_rasterized_driveable_area(scene.city_name)

    # Get 3D transform from camera to world coordinates
    extrinsic = scene.get_calibration(camera).extrinsic
    pose = scene.get_pose(frame).transform_matrix
    cam_to_world_tfm = np.matmul(pose, np.linalg.inv(extrinsic))

    # Get 2D affine transform from camera to map coordinates
    cam_to_map_tfm = np.matmul(map_tfm, cam_to_world_tfm[[0, 1, 3]])
    
    # Get 2D affine transform from BEV coords to map coords
    bev_to_cam_tfm = np.array([[resolution, 0, x1], 
                               [0, resolution, z1], 
                               [0, 0, 1]])
    bev_to_map_tfm = np.matmul(cam_to_map_tfm[:, [0, 2, 3]], bev_to_cam_tfm)

    # Warp map image to bev coordinate system
    mask = affine_transform(city_mask, bev_to_map_tfm[[1, 0]], 
                            output_shape=(mask_width, mask_height)).T
    return mask[None]



    








