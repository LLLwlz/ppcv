import os
from pylab import *
import cv2
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']

img = cv2.imread('test.jpg', 0)#转化为灰度图
img_color = img
blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
canny = cv2.Canny(blur, 100, 150)  # 100是最小阈值,150是最大阈值
# cv2.namedWindow("canny",cv2.WINDOW_NORMAL);#可调大小
# cv2.namedWindow("test", cv2.WINDOW_NORMAL);#可调大小
# cv2.imshow('test', img)
# cv2.imshow('canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(canny, cmap=plt.cm.gray)
axes[1].set_title("canny检测后结果")
plt.show()