# import the necessary packages
import threading
from threading import Thread

import imutils
import numpy as np
import cv2
import time
import csv
import schedule

filename = 'image_LD.png'
lower_range = np.array([0,100,100], dtype=np.uint8)
upper_range = np.array([10, 255, 255], dtype=np.uint8)

lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])

# load the image, convert it to grayscale, and blur it
initfunc = False
notdetectedvar = False
global data

def job():
    print("CSV Updated")
    global data
    with open('Level_Detect.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()


def i_run_once():
    global initfunc
    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)
    if initfunc == False:
        for i in range(50):
            ret, imgg = cap.read()
            imgg = imutils.resize(imgg, width=700)
            file = "image_LD.png"
            cv2.imwrite(file, imgg)
            cv2.putText(imgg, 'place level indicator in front of camera', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow('cam', imgg)
            time.sleep(0.1)
            k = cv2.waitKey(10)
            if k == 27:
                break
            time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()
    initfunc = False

def main():

    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)

    while (cap.isOpened()):
        global data
        ret, imgg = cap.read()
        imgg = imutils.resize(imgg, width=500)
        cv2.imshow('cam', imgg)
        image = imutils.resize(imgg, width=400)
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sigma = 0.33
        v = np.median(grayScale)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))

        edged = cv2.Canny(grayScale, low, high)

        # After finding edges we have to find contours
        # Contour is a curve of points with no gaps in the curve
        # It will help us to find location of shapes

        # cv2.RETR_EXTERNAL is passed to find the outermost contours (because we want to outline the shapes)
        # cv2.CHAIN_APPROX_SIMPLE is removing redundant points along a line
        (_, cnts, _) = cv2.findContours(edged,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # compute the moment of contour
            M = cv2.moments(c)
            # From moment we can calculte area, centroid etc
            # The center or centroid can be calculated as follows
            # print(M['m00'])
            if M['m00'] == 0:
                break
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            area = cv2.contourArea(c)
            # print('area : ', area)
            perimeter = cv2.arcLength(c, True)
            # print('peremeter : ',perimeter)
            # Outline the contours
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)  # offsets - with this you get 'mask'
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('cutted contour', image[y:y + h, x:x + w])
            crop_img = image[y:y + h, x:x + w]
            # show the output image

            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            mask0 = cv2.inRange(hsv, lower_range, upper_range)
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            size = area - perimeter
            # print('size ', size)
            mask = mask0 + mask1
            PixelsInRange = cv2.countNonZero(mask)
            div = PixelsInRange + perimeter
            # print('div ', div)
            frac_red = np.divide(float(div), int(size))
            percent_red = np.multiply((float(frac_red)), 100)
            if percent_red > 100:
                percent_red  = 100
            if percent_red < 0:
                break

            print('Level : ' + str(int(percent_red)) + '%')
            data = [int(percent_red)]

        schedule.run_pending()
        k = cv2.waitKey(10)
        if k == 27:
            break

if __name__=='__main__':
    i_run_once()
    schedule.every(10).seconds.do(job)
    main()
