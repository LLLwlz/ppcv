import sys
import cv2
import numpy as np

sys.path.insert(1, '../')
import pykinect_azure as pykinect

img_point = np.empty(shape=[0, 2], dtype=int)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global img_point
        img_point = np.append(img_point, [[x, y]], axis=0)


if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
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
        cv2.setMouseCallback("Color Image", on_EVENT_LBUTTONDOWN)

        key = cv2.waitKey(1)

        # Press Esc key to stop
        if key == 27:
            break

        # Press space key to save
        if key == 32:
            if np.size(img_point, 0) != 0:
                cv2.drawMarker(color_img, (img_point[-1, 0], img_point[-1, 1]), (255, 0, 0),
                               markerType=cv2.MARKER_TRIANGLE_DOWN)
            cv2.imwrite('Color_image.png', color_img)
            continue

        if key == 13:
            # Get row number
            print(np.size(img_point, 0), ' click:')
            # Get color_point_2d
            color_point_2d = pykinect.k4a._k4atypes.k4a_float2_t()
            color_point_2d.xy.x = img_point[-1, 0]
            color_point_2d.xy.y = img_point[-1, 1]
            print('color_point_2d\'s x:', color_point_2d.xy.x)
            print('color_point_2d\'s y:', color_point_2d.xy.y)

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
            point_3d = device.calibration.convert_2d_to_3d(depth_point_2d, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH,
                                                           pykinect.K4A_CALIBRATION_TYPE_DEPTH)
            print('point_3d\'s x', point_3d.xyz.x)
            print('point_3d\'s y', point_3d.xyz.y)
            print('point_3d\'s z', point_3d.xyz.z)
            img_path = 'image/Color_image_' + str(np.size(img_point, 0)) + '.png'
            cv2.drawMarker(color_img, (img_point[-1, 0], img_point[-1, 1]), (255, 0, 0),
                           markerType=cv2.MARKER_TRIANGLE_DOWN)
            cv2.imwrite(img_path, color_img)
