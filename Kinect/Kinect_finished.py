import sys

import cv2
import pykinect_azure as pykinect
from paddleocr import PaddleOCR

sys.path.insert(1, '../')
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复副本冲突问题

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True

    # Start device
    device = pykinect.start_device(config=device_config)

    # print(device_config.__str__())

    cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
    while True:

        # Get capture
        capture = device.update()

        # Get the depth image from the capture
        depth_ret, depth_img = capture.get_depth_image()

        # Get the color depth image from the capture
        color_ret, color_img = capture.get_color_image()

        if not depth_ret:
            continue

        if not color_ret:
            continue

        # Plot the image
        cv2.imshow('Color Image', color_img)

        key = cv2.waitKey(1)

        # Press Esc key to stop
        if key == 27:
            break

        # Press space key to save
        if key == 32:
            cv2.imwrite('Color_image.png', color_img)
            # continue

            # enable ocr recognizer
            ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,
                            lang="ch")  # need to run only once to download and load model into memory
            result = ocr.ocr('Color_image.png', cls=True)
            boxes = [line[0] for line in result]
            result_string = [line[1][0] for line in result]
            print(result_string)

            # match target string
            tar_string = input("Please put target string")
            if tar_string == "":
                print("Don't want to get any target string")
                continue

            index = len(result_string)
            for i in range(len(result_string)):
                if tar_string in result_string[i]:
                    index = i
                    break

            if index == len(result_string):
                print("Don't find the target string")
                continue

            location = boxes[index]

            for i in range(4):
                # Get color_point_2d
                color_point_2d = pykinect.k4a._k4atypes.k4a_float2_t()
                color_point_2d.xy.x = int(location[i][0])
                color_point_2d.xy.y = int(location[i][1])
                # print('color_point_2d\'s x:', color_point_2d.xy.x)
                # print('color_point_2d\'s y:', color_point_2d.xy.y)

                # Chage color_point_2d to depth_point_2d
                hadle_depth_img = capture.get_depth_image_object().handle()
                depth_point_2d = device.calibration.convert_color_2d_to_depth_2d(color_point_2d, hadle_depth_img)
                # print('depth_point_2d\'s x:', depth_point_2d.xy.x)
                # print('depth_point_2d\'s y:', depth_point_2d.xy.y)

                # Get depth data from depth_image
                # depth = depth_img[int(depth_point_2d.xy.x), int(depth_point_2d.xy.y)]
                found = False
                window_size = 1
                depth = 0
                height = device.calibration._handle.color_camera_calibration.resolution_height
                width = device.calibration._handle.color_camera_calibration.resolution_width
                while found == False:
                    window_size *= 2
                    step = window_size / 2
                    x_lower = max(depth_point_2d.xy.x - step, 0)
                    x_upper = min(depth_point_2d.xy.x + step, height)

                    y_lower = max(depth_point_2d.xy.y - step, 0)
                    y_upper = min(depth_point_2d.xy.y + step, width)
                    value = 0
                    number = 0
                    for x in range(int(x_lower), int(x_upper)):
                        for y in range(int(y_lower), int(y_upper)):
                            if depth_img[int(depth_point_2d.xy.x), int(depth_point_2d.xy.y)] > 0:
                                found = True
                                value += depth_img[int(depth_point_2d.xy.x), int(depth_point_2d.xy.y)]
                                number += 1
                    if found == True:
                        depth = value / number

                    if window_size > 128:
                        print(1)
                        break
                if found == False:
                    print('Could not get current depth')
                    print(depth)
                    continue
                # print('depth:', depth)

                # Chage 2D to 3D
                point_3d = device.calibration.convert_2d_to_3d(depth_point_2d, depth,
                                                               pykinect.K4A_CALIBRATION_TYPE_DEPTH,
                                                               pykinect.K4A_CALIBRATION_TYPE_DEPTH)
                # print('point_3d\'s x', point_3d.xyz.x)
                # print('point_3d\'s y', point_3d.xyz.y)
                # print('point_3d\'s z', point_3d.xyz.z)
                print(
                    "The target book's location-{0} point: point_2d is [{1}, {2}], point_3d is [{3}, {4}, {5}]".format(
                        i, color_point_2d.xy.x, color_point_2d.xy.y, point_3d.xyz.x, point_3d.xyz.y, point_3d.xyz.z))
