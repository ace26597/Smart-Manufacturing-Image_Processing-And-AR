# import the necessary packages
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
cap = cv2.VideoCapture(0)
if cap.read()[0] == False:
    cap = cv2.VideoCapture(1)

def notdetected():
    global notdetectedvar
    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)
    if notdetectedvar == False:
        for i in range(50):
            ret, imgg = cap.read()
            imgg = imutils.resize(imgg, width=700)
            file = "image_LD.png"
            cv2.imwrite(file, imgg)
            cv2.putText(imgg, 'place level indicator in front of camera', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
            cv2.imshow('cam', imgg)
            time.sleep(0.1)
            k = cv2.waitKey(10)
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    main()

def i_run_once():
    global initfunc
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
    initfunc = True

def ping():
    print("CSV Updated")
    with open('Level_Detect.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def run_threaded(ping):
    job_thread = Thread(target=ping)
    job_thread.start()

schedule.every(10).seconds.do(run_threaded, ping)

def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = cap.read()
 file = "image_LD.png"
 cv2.imwrite(file, im)
 return im

def main():
    cap = cv2.VideoCapture(0)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(1)

    while (cap.isOpened()):
        ret, imgg = cap.read()
        imgg = imutils.resize(imgg, width=500)
        cv2.imshow('cam', imgg)
        k = cv2.waitKey(10)
        if k == 27:
            break
        get_image()
        filename = 'image_LD.png'
        img = cv2.imread('image_LD.png')
        schedule.every(10).seconds.do(run_threaded, ping)
        ret, image = cap.read()
        image = imutils.resize(image, width=400)
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find edges in the image using canny edge detection method
        # Calculate lower threshold and upper threshold using sigma = 0.33
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
            #print(M['m00'])
            if M['m00'] == 0:
                print('Level Indicator Not Found \n 5 second window to place Level indicator in front of camera')
                cap.release()
                cv2.destroyAllWindows()
                notdetectedvar = False
                notdetected()

            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            area = cv2.contourArea(c)
            #print('area : ', area)
            perimeter = cv2.arcLength(c, True)
            #print('peremeter : ',perimeter)
            # Outline the contours
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)  # offsets - with this you get 'mask'
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow('cutted contour', image[y:y + h, x:x + w])
            crop_img = image[y:y + h, x:x + w]
            # show the output image

            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            mask0 = cv2.inRange(hsv, lower_range, upper_range)
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            size = crop_img.size / 3
            mask = mask0 + mask1
            blurred = cv2.GaussianBlur(mask, (11, 11), 0)
            PixelsInRange = cv2.countNonZero(mask)
            #print("PixelsInRange", PixelsInRange)
            frac_red = np.divide(float(PixelsInRange), int(size))
            #print(frac_red)
            percent_red = np.multiply((float(frac_red)), 100)
            print('Level : ' + str(percent_red) + '%')
            #cv2.imshow('mask',mask)
            data = [[percent_red]]

        cv2.waitKey(0)
        schedule.run_pending()

if __name__=='__main__':
    i_run_once()
    main()
    schedule.every(10).seconds.do(run_threaded, ping)
