import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
point_rend_metadata = MetadataCatalog.get("cityscapes_fine_sem_seg_val")
print(point_rend_metadata.stuff_classes)


# import PointRend project
from detectron2.projects import point_rend

im = cv2.imread("data/argo/argoverse-tracking/train/10b8dee6-778f-33e4-a946-d842d2d9c3d7/ring_front_center/ring_front_center_315968229010581848.jpg")
cv2.imwrite('test.jpg', im)


cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/tomb/detectron2_repo/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "/project_data/dolan/tomb/dean_models/model_final_cf6ac1.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs["sem_seg"].shape)

seg_mask = torch.argmax(outputs["sem_seg"], dim=0)
np.save("sem_test", seg_mask.detach().cpu().numpy())
print(seg_mask.shape)
print(point_rend_metadata.stuff_classes)
print(type(point_rend_metadata.stuff_classes))
v = Visualizer(im[:, :, ::-1], point_rend_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
point_rend_result = v.draw_sem_seg(seg_mask.to("cpu")).get_image()
cv2.imwrite('sem_test.jpg', point_rend_result[:, :, ::-1])