# -*- encoding: utf-8 -*-
'''
@File    :   img.py
@Time    :   2020/12/10 16:47:42
@Author  :   陆天天
@Version :   1.0
@Contact :   18857917788@163.com
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('code//img2.png')
plt.imshow(img)
mask = np.zeros((img.shape[:2]), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (10, 10, 950, 810)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('original image')
plt.subplot(1,2,2)
img = img * mask2[:, :, np.newaxis]
plt.imshow(img)
plt.title('target image')
plt.show()

max_corners = 10
quality_level = 0.01
min_distance = 10.0

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresholdValue, image_gray = cv2.threshold(imgray, 0, 255, 0)
image_gray = cv2.GaussianBlur(image_gray,(3,3),1)
corners = cv2.goodFeaturesToTrack(image_gray,
                                  max_corners,
                                  quality_level,
                                  min_distance,
                                  )
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
img_gauss = cv2.GaussianBlur(image_gray,(3,3),1)
plt.imshow(img)
plt.show()
