import cv2

import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 20, 0)
time.sleep(2.2)

w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0

# Model
import torch
model = torch.hub.load(   'yolov5' # Use backend for yolov5 in this folder
                        , 'custom' # to use model in this folder
                        , path='best.pt' # the name of model is this folder ## HERE
                        , source='local' # to use backend from this folder
                        , force_reload=True # clear catch
                        , device = 'cpu' # I want to use CPU
                    ) 
model.conf = 0.25 # NMS confidence threshold
model.iou = 0.45  # IoU threshold
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

def findbot(img):
    # Preprocess the image for the YOLOv5 model
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Use the YOLOv5 model to predict the bounding boxes for faces in the image
    preds = model.predict(img)

    # Extract the bounding boxes and confidence scores from the model's output
    boxes = []
    scores = []
    for pred in preds:
        for b in pred:
            box = b[:4] * 512
            score = b[4]
            boxes.append(box)
            scores.append(score)

    # Filter the bounding boxes based on their confidence scores
    boxes = [box for i, box in enumerate(boxes) if scores[i] > 0.5]

    # Draw the bounding boxes on the image
    for box in boxes:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Extract the center points and areas of the bounding boxes
    mybotListC = []
    mybotListArea = []
    for box in boxes:
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        mybotListC.append([cx, cy])
        mybotListArea.append(area)

    # Return the modified image and the face information
    if len(mybotListArea) != 0:
        i = mybotListArea.index(max(mybotListArea))
        return img, [mybotListC[i], mybotListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackbot(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    # print(speed, fb)
    me.send_rc_control(0, fb, 0, speed)

    return error

# cap = cv2.VideoCapture(1)

while True:
    # _, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findbot(img)
    pError = trackbot(info, w, pid, pError)
    # print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        me.land()
        break