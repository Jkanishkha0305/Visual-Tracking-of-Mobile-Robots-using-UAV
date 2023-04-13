#working
from djitellopy import tello
import KeyPressModule as kp
import time
import cv2
import cvzone

thres = 0.50
nmsThres = 0.2

classNames = []
classFile = 'ss.names' # Contains a totoal of 91 different objects which can be recognized by the code
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

print(classNames)
configPath = 'yolov5test1.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
global img
me.streamoff()
me.streamon()

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed
    if kp.getKey("q"):
        me.land();
        time.sleep(3)
    if kp.getKey("e"):
        me.takeoff()
    if kp.getKey("z"):
        cv2.imwrite(f'Test/Images/{time.time()}.jpg', img)
        time.sleep(0.3)
    return [lr, fb, ud, yv]

while True:
    # success, img = cap.read()
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres,
                                       nmsThreshold=nmsThres)  # To remove duplicates / declare accuracy
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)