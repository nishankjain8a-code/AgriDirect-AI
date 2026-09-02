from ultralytics import YOLO
model=YOLO("yolo11n.pt")
model.train(data="dataset/data.yaml",epochs=60,imgsz=640,batch=16,patience=12,project="runs",name="crop_detector")
model.val(data="dataset/data.yaml")
print("Copy runs/crop_detector/weights/best.pt -> models/crop_detector.pt")
