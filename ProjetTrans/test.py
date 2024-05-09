import cv2
import numpy as np

# Load YOLO object detector
net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

DEFAULT_CONFIANCE = 0.5
THRESHOLD = 0.4

# Load COCO class labels
with open('coco.names', 'r') as f:
    LABELS = f.read().splitlines()

# Load image
frame = cv2.imread('assets/test2.jpg')
if frame is None:
    print("Error: Could not read the image.")
    exit()

height, width, _ = frame.shape

# Preprocess the image
blob = cv2.dnn.blobFromImage(frame, 1 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)  # Reduced input size
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes, confidences, classIDs = [], [], []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > DEFAULT_CONFIANCE:
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, W, H) = box.astype("int")
            x = int(centerX - (W / 2))
            y = int(centerY - (H / 2))
            boxes.append([x, y, int(W), int(H)])
            confidences.append(float(confidence))
            classIDs.append(classID)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)

np.random.seed(42)  # Pour la reproductibilité

if len(indexes) > 0:
    for i in indexes.flatten():
        if LABELS[classIDs[i]] in ['user', 'bag']:
            (x, y, w, h) = boxes[i]
            label = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            color = [int(c) for c in np.random.uniform(0, 255, size=(3,))]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 800, 600)
cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

cv2.destroyAllWindows()
