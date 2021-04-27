# NOTE: specific to community scenario
# NOTE: ctrashc means [community trash can]


import os
import numpy as np
import cv2
# import sys
# sys.path.extend([".", "..", "../..", "../../.."])

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo

from utils.predictor import VisualizationDemo
# from detectron2.engine import DefaultPredictor

setup_logger()

cfg = get_cfg()
# cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = os.path.join("/home/disk/checkpoints", "ctrashc_v21.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"

# cfg.INPUT.MAX_SIZE_TEST = 2000
# cfg.INPUT.MIN_SIZE_TEST = 512

cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 3000
# cfg.SOLVER.AMP.ENABLED = False

cfg.MODEL.DEVICE = 'cuda:2'

predictor = VisualizationDemo(cfg)
# predictor = DefaultPredictor(cfg)


def transform_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img


def get_prediction_ctrashc(image_bytes):
    im = transform_image(image_bytes=image_bytes)

    outputs, vis_outputs = predictor.run_on_image(im)

    pred_classes = outputs["instances"]._fields['pred_classes'].to("cpu").tolist()
    pred_boxes = outputs["instances"]._fields['pred_boxes'].to("cpu").tensor.tolist()
    scores = outputs["instances"]._fields['scores'].to("cpu").tolist()

    _cls, _x0, _y0, _x1, _y1 = [], [], [], [], []
    for idx, cls in enumerate(pred_classes):
        x0 = int(pred_boxes[idx][0])
        y0 = int(pred_boxes[idx][1])
        x1 = int(pred_boxes[idx][2])
        y1 = int(pred_boxes[idx][3])

        _cls.append(cls)
        _x0.append(x0)
        _x1.append(x1)
        _y0.append(y0)
        _y1.append(y1)

    return _cls, _x0, _y0, _x1, _y1
