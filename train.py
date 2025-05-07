from ultralytics import YOLO

model = YOLO("yolov8m.pt")  

model.train(
    data="data.yaml",
    epochs=200,
    imgsz=640
)
