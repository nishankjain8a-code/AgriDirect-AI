from ultralytics import YOLO
import sys
model=YOLO("models/crop_detector.pt")
model.predict(source=sys.argv[1] if len(sys.argv)>1 else "sample.jpg",conf=0.35,save=True)
