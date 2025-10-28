## WORKSHOP-2-Object-detection-using-web-camera

### NAME: AMRUTHAVARSHINI GOPAL
### REGISTER NUMBER: 212223230013

### AIM :

To perform real-time object detection using a trained YOLO v4 model through your laptop camera.

### ALGORITHM :

1)Load YOLO model and class files.

2)Start webcam capture.

3)Convert frame to blob and feed to model.

4)Get detections and draw boxes.

5)Show output until ‘q’ is pressed.


### PROGRAM :
```
# YOLOv4 Real-Time Detection in Jupyter (Smooth Display + Fixed NMS)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import time

# ---------------------------
# Step 1: Load YOLOv4
# ---------------------------
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = open("coco.names").read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ---------------------------
# Step 2: Open Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

# ---------------------------
# Step 3: Detection Loop
# ---------------------------
plt.ion()  # Interactive mode for continuous update
fig, ax = plt.subplots(figsize=(8,6))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Create blob and feed to YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Collect boxes, confidences, class IDs
        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (Fixed)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indices) > 0:
            indices = np.array(indices).reshape(-1)  # Flatten safely
            for i in indices:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                conf = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert BGR -> RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame in Jupyter
        ax.clear()
        ax.imshow(frame_rgb)
        ax.axis('off')
        display(fig)
        clear_output(wait=True)

        time.sleep(0.03)  # Reduce flickering

except KeyboardInterrupt:
    print("Stopped by user")

# ---------------------------
# Step 4: Release Resources
# ---------------------------
cap.release()
plt.close()
```

## OUTPUT:

<img width="971" height="720" alt="Screenshot 2025-10-14 162235" src="https://github.com/user-attachments/assets/a1ce1083-9423-44be-ad54-392c6f5c73fe" />



## RESULT:

The real-time object detection using a trained YOLO v4 model through your laptop camera is executed and performed successfully.
