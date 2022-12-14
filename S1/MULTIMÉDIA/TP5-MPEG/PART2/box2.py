import cv2
import numpy as np
from math import inf
import time


def MSE(bloc1, bloc2):
    block1, block2 = np.array(bloc1), np.array(bloc2)
    return np.square(np.subtract(block1, block2)).mean()


img1 = cv2.imread('images/image072.png')
img2 = cv2.imread('images/image092.png')

grayImg1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

height = grayImg1.shape[0]
width = grayImg2.shape[1]

boxSize = 16

greens = []
reds = []
start_time = time.time()
for i in range(0, height - boxSize, boxSize):  # colonne
    for j in range(0, width - boxSize, boxSize):  # ligne
        blocRouge = grayImg1[i:i + boxSize, j:j + boxSize]
        min1 = inf
        for i1 in range(max(0, i - 7), min(i + 7, height - boxSize)):
            for j1 in range(max(0, j - 7), min(j + 7, width - boxSize)):
                blocVert = grayImg2[i1:i1 + boxSize, j1:j1 + boxSize]

                loss = MSE(blocRouge, blocVert)
                if loss < min1:
                    min1 = loss
                    x1 = i1
                    x2 = j1

        if min1 > 50:
            # print(min1)
            greens.append((x2, x1))
            reds.append((i, j))

for i in range(len(greens)):
    cv2.rectangle(img2, (greens[i][0], greens[i][1]),
                  (greens[i][0] + boxSize, greens[i][1] + boxSize), (0, 255, 0), 2)

for i in range(len(reds)):
    cv2.rectangle(img1, (reds[i][0], reds[i][1]),
                  (reds[i][0] + boxSize, reds[i][1] + boxSize), (0, 0, 255), 2)

time = time.time() - start_time
print(f"{time} seconds")

cv2.imshow('image1', img1)
cv2.imshow('image2', img2)


cv2.waitKey(0)
