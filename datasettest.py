#working
from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
frame_read = tello.get_frame_read()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imwrite("     picture{}.png",img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)