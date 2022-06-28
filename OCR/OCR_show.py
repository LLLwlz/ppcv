from paddleocr import PaddleOCR, draw_ocr
# 显示结果
from PIL import Image
import cv2
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == '__main__':

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="ch")  # need to run only once to download and load model into memory
    img_path = 'test.jpg'
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)
    # boxes=np.array([line[0] for line in result],np.int32)
    # print(boxes)
    img=cv2.imread("test.jpg",1)
    i=0
    # cv2.polylines(img, [boxes], isClosed=True, color=(0, 0, 255), thickness=1)
    for line in result:
        box=np.array([line[0]],np.int32)
        print(box)
        # box.reshape((-1,-1,2))
        cv2.polylines(img, [box], isClosed=True, color=(0, 0, 255), thickness=8)
        # print(type(box[0][1]))
        cv2.putText(img,str(i),tuple(box[0][0]),cv2.FONT_ITALIC,3,(0,0,0),2)
        i=i+1
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()