import cv2
import numpy as np
import HandTrackingModule as htm
import time
import datetime
import pyautogui
import csv


#############################################
wCam, hCam = 640, 480
wscr, hscr = 1920, 1080
frameR = 100  # size of the box for frame in which mouse will move
smoothening = 7
pTime = 0
plocx, plocy = 0, 0  # Previous location of x and y
clocx, clocy = 0, 0  # Current location of x and y
############################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # 4. Only Index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Co-ordinates

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wscr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hscr))
            # 6. Smothen Values
            clocx = plocx + (x3 - plocx)/smoothening
            clocy = plocy + (y3 - plocy)/smoothening
            # 7. Move Mouse
            pyautogui.moveTo(clocx, clocy)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy
        # 8. Both Index and middle fingers are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineinfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 30:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)

                # 10. Click mouse if distance short
                pyautogui.click()
                time_storage = datetime.datetime.now()
                print(time_storage.strftime("%H:%M"))

    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow('image', img)
    cv2.waitKey(1)
