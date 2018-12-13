# Import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

from imutils import paths
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "path to the image file")
# args = vars(ap.parse_args())
basePath = r"images"
imagePaths = list(paths.list_images(basePath))
output1 = []
output_final = []
columns = 3
for (i, imagePath) in enumerate(imagePaths):

    # load the image and convert it to grayscale
    # image = cv2.imread(args['image'])
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow('Gradient', gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    # cv2.imshow('Blurred', blurred)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Thresh', thresh)
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Closed', closed)
    # perform the series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    # cv2.imshow('Closed2', closed)

    # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.convenience.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    # print("box: ", box)
    box = np.int0(box)
    # print("box (after int0): ", box)

    # draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)

    output1.append(image)
    if (i+1) % columns == 0:
        output_final.append(np.hstack(output1))
        output1 = []
    cv2.waitKey(0)

output_final2 = np.vstack(output_final)
cv2.imshow("output", output_final2)
cv2.waitKey(0)
import os
outpath = os.path.join(basePath, "final.jpg")
cv2.imwrite(outpath, output_final2)


