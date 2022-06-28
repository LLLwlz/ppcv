import os
from pylab import *
import cv2
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']

img = cv2.imread('test.jpg', 0)#转化为灰度图
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
Scale_absY = cv2.convertScaleAbs(y)
sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
# cv2.namedWindow("canny",cv2.WINDOW_NORMAL);#可调大小
# cv2.namedWindow("test", cv2.WINDOW_NORMAL);#可调大小
# cv2.imshow('test', img)
# cv2.imshow('canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(sobel, cmap=plt.cm.gray)
axes[1].set_title("sobel检测后结果")
plt.show()