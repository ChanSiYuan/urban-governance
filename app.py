import time
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request

from objects.can.ctrashc import get_prediction_ctrashc
from objects.can.strashc import get_prediction_strashc
from objects.trash import get_prediction_trash
from objects.truck import get_prediction_truck
from objects.flotage import get_prediction_flotage
from objects.blot import get_prediction_blot


app = Flask(__name__)

time_fmt = "%Y-%m-%d %H:%M:%S"


@app.route("/predict_ctrashc", methods=["POST"])
def predict_ctrashc():
    """
    The api for community trash can detection task;
    Detect objects contain: mobile trash can (label: mobile)
                            trash bag (label: bag)
    """
    if request.method == "POST":
        time_start = time.time()
        file = request.files["file"]
        img_bytes = file.read()
        preds = get_prediction_ctrashc(image_bytes=img_bytes)
        _cls, _x0, _y0, _x1, _y1, = preds
        time_end = time.time()
        curr_time = time.strftime(time_fmt, time.localtime())
        print(f"[{curr_time}] inference [can-ctrashc] time cost: {time_end - time_start:2f} s")
        return jsonify({
            "cls": _cls,
            "x0": _x0,
            "y0": _y0,
            "x1": _x1,
            "y1": _y1
        })


@app.route('/predict_can', methods=['POST'])
def predict_can():
    """
    The api for street trash can detection task;
    Detect objects contain: fixed trash can
    """
    if request.method == 'POST':
        time_start = time.time()
        file = request.files['file']
        img_bytes = file.read()
        preds = get_prediction_strashc(image_bytes=img_bytes)
        _cls, _x0, _y0, _x1, _y1, = preds
        time_end = time.time()
        curr_time = time.strftime(time_fmt, time.localtime())
        print(f"[{curr_time}] inference [can-strashc] time cost: {time_end - time_start:2f} s")
        return jsonify({
            'cls': _cls,
            'x0': _x0,
            'y0': _y0,
            'x1': _x1,
            'y1': _y1
        })


@app.route('/predict', methods=['POST'])
def predict():
    """
    The api for street truck detection task, and recognize whether the truck clean or not;
    Detect object contain: truck
    :return:
    """
    if request.method == 'POST':
        time_start = time.time()
        file = request.files['file']
        img_bytes = file.read()
        preds = get_prediction_truck(image_bytes=img_bytes)
        _cls, _x0, _y0, _x1, _y1, _ratio, _neat = preds
        time_end = time.time()
        curr_time = time.strftime(time_fmt, time.localtime())
        print(f"[{curr_time}] inference [truck] time cost: {time_end - time_start:2f} s")
        return jsonify({'cls': _cls,
                        'ratio': _ratio,
                        'x0': _x0,
                        'y0': _y0,
                        'x1': _x1,
                        'y1': _y1,
                        'neat': _neat})


@app.route('/predict_smpfw', methods=['POST'])
def predict_smpfw():
    """
    The api for flotages detection that flowing on the river.
    Dtection object contain: flotage
    :return:
    """
    if request.method == 'POST':
        time_start = time.time()
        file = request.files['file']
        img_bytes = file.read()
        preds = get_prediction_flotage(image_bytes=img_bytes)
        _cls, _x0, _y0, _x1, _y1, = preds
        time_end = time.time()
        curr_time = time.strftime(time_fmt, time.localtime())
        print(f"[{curr_time}] inference [smpfw] time cost: {time_end - time_start:2f} s")
        return jsonify({
            'cls': _cls,
            'x0': _x0,
            'y0': _y0,
            'x1': _x1,
            'y1': _y1
        })

