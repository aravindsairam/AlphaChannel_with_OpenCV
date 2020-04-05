import numpy as np
import argparse
import dlib
import cv2

# python Transparent.py --img images/image_name
parser = argparse.ArgumentParser(description = 'User input')
parser.add_argument('--img', help = 'Input image name')
args = parser.parse_args()

image = cv2.imread(args.img)
image = cv2.resize(image, None, fx = 1/2, fy = 1/2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect predcited rectangle with dlib
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1)

temp = np.zeros(image.shape, dtype = np.uint8)

for (i, rect) in enumerate(rects):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), -1)

temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
masked = cv2.bitwise_and(image, image, mask = temp)

#convert image to RGB to RGBA channel
img = cv2.cvtColor(masked, cv2.COLOR_RGB2RGBA)

i, j, k = img.shape

#Run through all pixels and
#if a pixel is black make it transparent by setting alpha channel = 0
for a in range(0, i):
    for b in range(0, j):
        if img[a, b][0] == 0 and img[a, b][1] == 0 and img[a, b][2] == 0:
            img[a, b][3] = 0

cv2.imwrite('images/transparent.png', img)
