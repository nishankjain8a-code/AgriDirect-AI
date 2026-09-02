# Custom Crop Detector

This prototype contains the complete YOLO training + inference path. A custom accuracy figure must come from labelled field data and held-out validation; no metric is fabricated.

## Dataset
YOLO format:
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
dataset/data.yaml

Label: `class_id x_center y_center width height` normalized 0–1.

Recommended classes: Tomato, Onion, Potato, Chilli, Wheat, Cotton, Unknown.

## Train
`pip install -r requirements-train.txt`
`python train.py`

Copy `runs/crop_detector/weights/best.pt` to `models/crop_detector.pt`.

The Streamlit app then shows crop bounding boxes + confidence and the ORBIT Scan Agent uses detector confidence to guide the farmer.
