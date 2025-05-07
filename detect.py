import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Make predictions
    results = model(frame)

    # If results are detected, render the results (bounding boxes, labels, etc.)
    if results[0].boxes.shape[0] > 0:  # Check if there are any detected objects
        for result in results[0].boxes.data:  # Access the first result's boxes
            # Get the class index and confidence
            conf = result[4].item()  # Confidence score
            label = model.names[int(result[5].item())]  # Class name

            # Draw bounding boxes and labels on the frame
            x1, y1, x2, y2 = map(int, result[:4])  # Extract bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Print detection details
            print(f"Detected: {label}, Confidence: {conf:.2f}")

    # Display the frame with the results
    cv2.imshow("Webcam - Areca Nut Detection", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
