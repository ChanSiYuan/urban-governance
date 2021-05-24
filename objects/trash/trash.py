import os
import numpy as np
import cv2

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from utils.predictor import VisualizationDemo

setup_logger()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("weights", "trash_v10.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.INPUT.MAX_SIZE_TEST = 512
cfg.MODEL.DEVICE = 'cuda:0'

predictor = VisualizationDemo(cfg)


def transform_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img


def get_prediction_trash(image_bytes):
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
