from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.val()  
mAP50 = results.metrics['mAP50']

if mAP50 > 0.90:
    print(f"Good job! Your model's mAP50 is {mAP50 * 100:.2f}% which is above 90.")
else:
    print(f"Your model's mAP50 is {mAP50 * 100:.2f}%. Consider improving the model.")
