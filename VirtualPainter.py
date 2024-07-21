import cv2
import numpy as np
import os
import HandTrackingModule as htm

##########################################
brushThickness = 15
eraserThickness = 100
#########################################

folderPath = "Header"
myList = os.listdir(folderPath)
overLayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
header = overLayList[0]

drawColor = (225, 0, 225)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # Clear canvas if all fingers are up
        if all(fingers):
            imgCanvas.fill(0)  # Efficient way to clear the canvas
            cv2.putText(img, "Canvas Cleared", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        # Selection mode: Two fingers are up
        elif fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overLayList[0]
                    drawColor = (225, 0, 225)
                elif 550 < x1 < 750:
                    header = overLayList[1]
                    drawColor = (225, 0, 0)
                elif 800 < x1 < 950:
                    header = overLayList[2]
                    drawColor = (225, 225, 0)
                elif 1050 < x1 < 1200:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)

        # Drawing mode: Index finger is up
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 225, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
