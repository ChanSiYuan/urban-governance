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
cfg.MODEL.WEIGHTS = os.path.join("weights", "truck_v10.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.INPUT.MAX_SIZE_TEST = 2000
cfg.MODEL.DEVICE = 'cuda:0'

predictor = VisualizationDemo(cfg)


def transform_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img

def check_truck_neat(truck_area, dirty_boxes):
    if dirty_boxes == []:
        return 0
    for idx, dirty_area in enumerate(dirty_boxes):
        if (truck_area[0] <= dirty_area[0] and
            truck_area[1] <= dirty_area[1] and
            truck_area[2] >= dirty_area[2] and
            truck_area[3] >= dirty_area[3]):
            return 1
    return 0

def get_prediction_truck(image_bytes):
    im = transform_image(image_bytes=image_bytes)
    h, w, _ = im.shape

    outputs, vis_outputs = predictor.run_on_image(im)
    pred_classes = outputs["instances"]._fields['pred_classes'].to("cpu").tolist()
    pred_boxes = outputs["instances"]._fields['pred_boxes'].to("cpu").tensor.tolist()
    scores = outputs["instances"]._fields['scores'].to("cpu").tolist()

    dirty_inds, dirty_boxes = [], []
    for i in range(len(pred_classes)):
        if pred_classes != 2:
            continue
        dirty_inds.append(i)

        x0 = int(pred_boxes[i][0])
        y0 = int(pred_boxes[i][1])
        x1 = int(pred_boxes[i][2])
        y1 = int(pred_boxes[i][3])
        dirty_boxes.append([x0, y0, x1, y1])

    _cls, _neat, _ratio, _x0, _y0, _x1, _y1 = [], [], [], [], [], [], []

    for idx, cls in enumerate(pred_classes):
        if scores[idx] < 0.7 or cls == 2:
            continue

        x0 = int(pred_boxes[idx][0])
        y0 = int(pred_boxes[idx][1])
        x1 = int(pred_boxes[idx][2])
        y1 = int(pred_boxes[idx][3])

        area_size = (y1 - y0) * (x1 - x0)
        area_ratio = area_size / (h * w)
        if cls == 1:
            _cls.append(cls)
            _x0.append(x0)
            _y0.append(y0)
            _x1.append(x1)
            _y1.append(y1)
            _neat.append(-1)
            _ratio.append(area_ratio)
        else:
            _cls.append(cls)
            _x0.append(x0)
            _y0.append(y0)
            _x1.append(x1)
            _y1.append(y1)
            truck_area = [x0, y0, x1, y1]
            _neat.append(check_truck_neat(truck_area, dirty_boxes))
            _ratio.append(area_ratio)

    return _cls, _x0, _y0, _x1, _y1, _ratio, _neat