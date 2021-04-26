import os
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

from utils.predictor import VisualizationDemo
from .classification import make_model, cfg as opt

__all__ = ["get_prediction_truck"]


class Classifier(object):
    def __init__(self):

        opt.merge_from_file("./configs/Classification/R_50_2fc.yaml")
        opt.freeze()

        self.device = torch.device(opt.DEVICE)
        # model
        model = make_model(opt, num_classes=2)
        model.load(path=os.path.join("/home/disk/checkpoints", "ResModel_best.pth"))
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def transform_image(image):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])

        return my_transforms(Image.fromarray(image)).unsqueeze(0)

    def __call__(self, img):
        with torch.no_grad():
            img = self.transform_image(img)
            img = img.to(self.device)
            score = self.model(img)
            target = np.argmax(score.cpu().numpy(), 1)

        # 0:neat 1:non-neat
        return target[0]


setup_logger()

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join("/home/disk/checkpoints", "model_final_truck_other.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.MODEL.DEVICE = 'cuda:0'

predictor = VisualizationDemo(cfg)
classifier = Classifier()


def transform_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img


def get_prediction_truck(image_bytes):
    im = transform_image(image_bytes=image_bytes)
    h, w, _ = im.shape
    area = h * w

    outputs, vis_outputs = predictor.run_on_image(im)
    pred_classes = outputs["instances"]._fields['pred_classes'].to("cpu").tolist()
    pred_boxes = outputs["instances"]._fields['pred_boxes'].to("cpu").tensor.tolist()
    scores = outputs["instances"]._fields['scores'].to("cpu").tolist()

    _cls, _x0, _y0, _x1, _y1, _ratio, _neat = [], [], [], [], [], [], []
    for idx, cls in enumerate(pred_classes):
        # 0:truck, 1:other
        if scores[idx] > 0.85:
            x0 = int(pred_boxes[idx][0])
            y0 = int(pred_boxes[idx][1])
            x1 = int(pred_boxes[idx][2])
            y1 = int(pred_boxes[idx][3])

            tmp = (y1 - y0) * (x1 - x0)
            ratio = tmp / area

            _cls.append(cls)
            _x0.append(x0)
            _x1.append(x1)
            _y0.append(y0)
            _y1.append(y1)
            _ratio.append(ratio)

            if cls == 0:
                im_box = im[y0:y1, x0:x1]
                neat = int(classifier(img=im_box))
                _neat.append(neat)
            else:
                _neat.append(-1)

    return _cls, _x0, _y0, _x1, _y1, _ratio, _neat
