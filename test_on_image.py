import numpy as np, cv2, imutils
from sklearn.externals import joblib
import warnings
import csv
from subprocess import check_output
import subprocess
from threading import Thread
import time
import schedule


def ping():
    print("done")
    with open('voltage.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(pred)
    csvFile.close()


def run_threaded(ping):
    job_thread = Thread(target=ping)
    job_thread.start()


schedule.every(10).seconds.do(run_threaded, ping)

warnings.filterwarnings("ignore")
img_counter = 0
# reading image
#img = cv2.imread('sample_image2.jpg')

cap = cv2.VideoCapture(0)
if cap.read()[0] == False:
    cap = cv2.VideoCapture(1)

while (cap.isOpened()):
    # resizing image
    ret, img = cap.read()

    img = imutils.resize(img, width=300)
    # converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # creating a kernel
    kernel = np.ones((50, 50), np.uint8)

    # applying blackhat thresholding
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # applying OTSU's thresholding
    ret, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # performing erosion and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # finding countours in image
    ret, cnts, hie = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loading our ANN model
    model = joblib.load('model.pkl')
    for c in cnts:
        try:
            # creating a mask
            mask = np.zeros(gray.shape, dtype="uint8")

            (x, y, w, h) = cv2.boundingRect(c)

            hull = cv2.convexHull(c)
            mask = cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)

            # Getting Region of interest
            roi = mask[y - 7:y + h + 7, x - 7:x + w + 7]
            roi = cv2.resize(roi, (28, 28))
            roi = np.array(roi)
            # reshaping roi to feed image to our model
            roi = roi.reshape(1, 784)

            # predicting
            prediction = model.predict(roi)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            pred = str(int(prediction[0])) + str(int(prediction[0]))+ '.' + str(int(prediction[0])) + 'V'
            cv2.putText(img, str(int(prediction)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        except Exception as e:
            None
            #print(e)

    img = imutils.resize(img, width=700)
    cv2.imshow('thresh',thresh)
    cv2.imshow('cam',img)
    k = cv2.waitKey(10)
    if k == 27:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
        break
    schedule.run_pending()

