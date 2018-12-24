# Import the necessary packages
import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import numpy as np
import ocr_template_match

refImagePath = r"images\\refFont.png"
digits = ocr_template_match.get_refDigits(refImagePath)

imagePath = r"images\\test2.jpg"
image = cv2.imread(imagePath)
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edged = cv2.Canny(blurred, 75, 250, 255)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# print("# of contours {} ".format(len(cnts)))
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

displayCnt = None
for (i, c) in enumerate(cnts):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # print("[INFO] {}): {}".format(i, len(approx)))
    if len(approx) == 4:
        # print("[INFO] {}): {}".format(i, len(approx)))
        image = cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        displayCnt = approx
        boundingBox = cv2.boundingRect(c)
        # cv2.imshow("Image1", image)
        # cv2.waitKey(0)

thresh_orig = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow("Thresh", thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh_orig_morph = cv2.morphologyEx(thresh_orig, cv2.MORPH_CLOSE, kernel)
thresh_orig_dilate = cv2.dilate(thresh_orig_morph, None, iterations=1)
# cv2.imshow("Thresh_orig", thresh_orig_dilate)

cnts_orig = cv2.findContours(thresh_orig_dilate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts_orig = imutils.grab_contours(cnts_orig)
# print("# of contours_orig {} ".format(len(cnts_orig)))

digitCnts = []
firstLine = []
secondLine = []
for c in cnts_orig:
    (x, y, w, h) = cv2.boundingRect(c)
    # print("[INFO] x: {}, y: {}, width: {}, height: {}".format(x, y, w, h))
    if w >= 25 and (h >= 30 and h <= 50) and x >= 100:
        # print("[INFO] x: {}, y: {}, width: {}, height: {}".format(x, y, w, h))
        if y < 250:
            firstLine.append(c)
        else:
            secondLine.append(c)

firstLine = contours.sort_contours(firstLine, method="left-to-right")[0]
secondLine = contours.sort_contours(secondLine, method="left-to-right")[0]

digitCnts = firstLine + secondLine

rects = [cv2.boundingRect(c) for c in digitCnts]
count = 0
firstNumber = []
secondNumber = []
for rect in rects:
    groupOutput = []
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 2)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1])
    pt2 = int(rect[0])
    x_len = int(rect[2])
    y_len = int(rect[3])
    roi = thresh_orig_dilate[pt1:(pt1+y_len), pt2:(pt2+x_len)]
    roi = cv2.resize(roi, (57, 88))
    # Initialize list of template matching scores
    scores = []
    # loop over reference digit name and digit ROI
    for (digit, digitROI) in digits.items():
        # apply correlation-based remplate matching, take the score, and update the scores list
        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        scores.append(score/1e8)
    # the classification for the digit ROI will be the reference digit name with the *largest* template matching score
    # print("[INFO] Scores: {}".format(scores))
    groupOutput.append(str(np.argmax(scores)))
    # groupOutput.append(str(np.argmin(scores)))
    # print(groupOutput)
    cv2.putText(image, groupOutput[0], (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    count += 1
    if count < 6:
        firstNumber.append(int(groupOutput[0]))
    else:
        secondNumber.append(int(groupOutput[0]))

finalNumber1 = str(firstNumber[0]) + "." + "".join(map(str, firstNumber[1:]))
finalNumber2 = str(secondNumber[0]) + "." + "".join(map(str, secondNumber[1:]))
print("[INFO] First Number: {}".format(finalNumber1))
print("[INFO] Second Number: {}".format(finalNumber2))


cv2.imshow("Result", image)
cv2.waitKey(0)
