#背景分割器
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorMOG2()
while (True):
    ret, frame = cap.read()
    fgmast = mog.apply(frame)
    th = cv2.threshold(fgmast.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    image, cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 1600:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('contours', frame)
    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

