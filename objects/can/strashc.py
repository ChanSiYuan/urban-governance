# NOTE: specific to street scenario
# NOTE: strashc means [street trash can]


import os
import numpy as np
import cv2
# import sys
# sys.path.extend([".", "..", "../..", "../../.."])

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

from utils.predictor import VisualizationDemo

setup_logger()

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join("/home/disk/checkpoints", "strashc_v20.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# cfg.INPUT.MAX_SIZE_TEST = 2000
# cfg.INPUT.MIN_SIZE_TEST = 512
cfg.MODEL.DEVICE = 'cuda:2'


predictor = VisualizationDemo(cfg)


def transform_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img


def get_prediction_strashc(image_bytes):
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
