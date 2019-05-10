from threading import Thread
import cv2
import imutils
import numpy as np
import csv
import schedule
import time

initfunc = False
min_angle = 60
max_angle = 300
min_value = 0
max_value = 240
units = 'psi'

def job():
    global data
    with open('gauge.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def i_run_once():
    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)
    global initfunc
    if initfunc == False:
        for i in range(50):
            #print("yep")
            ret, imgg = cap.read()
            imgg = imutils.resize(imgg, width=700)
            cv2.putText(imgg, 'place Analog Gauge in front of camera', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), 2)
            cv2.imshow('cam', imgg)
            k = cv2.waitKey(10)
            if k == 27:
                break
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    initfunc = True

def get_current_value(gray, min_angle, max_angle, min_value, max_value, x, y, r):
    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV);
    cv2.imshow('dst',dst2)
    minLineLength = 8
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=50, minLineLength=minLineLength,
                            maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    if lines is not None:
        final_line_list = []

        diff1LowerBound = 0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
        diff1UpperBound = 0.3
        diff2LowerBound = 0.1  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
        diff2UpperBound = 2.0

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
                # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp
                # check if line is within an acceptable range
                if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (
                        diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                    final_line_list.append([x1, y1, x2, y2])

        if len(final_line_list)<2:
            main()
        x1 = final_line_list[0][0]
        y1 = final_line_list[0][1]
        x2 = final_line_list[0][2]
        y2 = final_line_list[0][3]
        #cv2.line(dst2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imshow('gauge', dst2)
        # find the farthest point from the center to be what is used to determine the angle
        dist_pt_0 = dist_2_pts(x, y, x1, y1)
        dist_pt_1 = dist_2_pts(x, y, x2, y2)
        if (dist_pt_0 > dist_pt_1):
            x_angle = x1 - x
            y_angle = y - y1
        else:
            x_angle = x2 - x
            y_angle = y - y2
        # take the arc tan of y/x to find the angle
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        res = np.rad2deg(res)
        if x_angle > 0 and y_angle > 0:  # in quadrant I
            final_angle = 270 - res
        elif x_angle < 0 and y_angle > 0:  # in quadrant II
            final_angle = 90 - res
        elif x_angle < 0 and y_angle < 0:  # in quadrant III
            final_angle = 90 - res
        elif x_angle > 0 and y_angle < 0:  # in quadrant IV
            final_angle = 270 - res
        else:
            main()
        # print final_angle
        old_min = float(min_angle)
        old_max = float(max_angle)
        new_min = float(min_value)
        new_max = float(max_value)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        if new_value < new_max * 0.33:
            level = 'LOW'
        elif new_value < new_max * 0.66:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
    else:
        print('wtf')
    return new_value, level

def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    global data
    while (cap.isOpened()):
        ret, imgg = cap.read()
        cv2.imshow('cam', imgg)
        k = cv2.waitKey(10)
        if k == 27:
            break
        gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 400,
                                   param1=15, param2=40, minRadius=10, maxRadius=0)
        # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
        if circles is not None:
            a, b, c = circles.shape
            x, y, r = avg_circles(circles, b)
            val, level = get_current_value(gray, min_angle, max_angle, min_value, max_value, x, y, r)
            val = int(val)
            if val < 0 or val > 241:
                main()
            print("Current reading: %s %s" % (val, units))
            print("Gauge Reading Level : %s" % (level))
            data = [val, level]
        schedule.run_pending()
        k = cv2.waitKey(10)
        if k == 27:
            break

if __name__=='__main__':
    i_run_once()
    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)
    schedule.every(10).seconds.do(job)
    main()
    cap.release()
    cv2.destroyAllWindows()
