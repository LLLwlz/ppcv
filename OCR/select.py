from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 修复副本冲突问题

if __name__ == '__main__':

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="ch")  # need to run only once to download and load model into memory
    img_path = 'test.jpg'
    result = ocr.ocr(img_path, cls=True)

    for line in result:
        # print(type(line))

        print(line)
