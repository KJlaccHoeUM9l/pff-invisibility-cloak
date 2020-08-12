import cv2
import time
import numpy as np

from functools import reduce


class InvisibilityCloak:
    def __init__(self):
        self.__background = None
        self.__white_noise = None
        self.__camouflage1 = None
        self.__camouflage2 = None
        self.__last_background_key = ord('0')
        self.__inverse_mask = False

        self.__color_trackbar_image_window_name = 'Color_Trackbar'
        self.__lower_threshold = None
        self.__upper_threshold = None
        self.__lower_threshold_second = None
        self.__upper_threshold_second = None

        # Trackbars for color change
        cv2.namedWindow(self.__color_trackbar_image_window_name)
        cv2.createTrackbar('H', self.__color_trackbar_image_window_name, 10, 180, nothing)
        cv2.createTrackbar('S', self.__color_trackbar_image_window_name, 175, 255, nothing)
        cv2.createTrackbar('V', self.__color_trackbar_image_window_name, 210, 255, nothing)
        cv2.createTrackbar('H\nradius', self.__color_trackbar_image_window_name, 5, 15, nothing)
        cv2.createTrackbar('S\nradius', self.__color_trackbar_image_window_name, 70, 90, nothing)
        cv2.createTrackbar('V\nradius', self.__color_trackbar_image_window_name, 55, 90, nothing)

    def take_background_image(self, video_capture, time_delay=3, selection_size=5):
        print('Leave the scene, record of background image in {} sec'.format(time_delay))
        time.sleep(time_delay)

        print('Record of background image')
        for i in range(selection_size):
            _, self.__background = video_capture.read()
        self.__background = np.flip(self.__background, axis=1)

        image_size = (self.__background.shape[1], self.__background.shape[0])
        self.__white_noise = get_white_noise_image(self.__background.shape, 0, 255)
        self.__camouflage1 = cv2.resize(cv2.imread('3rdparty/backgrounds/1.jpg'), image_size)
        self.__camouflage2 = cv2.resize(cv2.imread('3rdparty/backgrounds/2.jpg'), image_size)

        print('Background has been taken, return to scene')
        print('Resumption of work in {} sec'.format(time_delay))
        time.sleep(time_delay)

    def update_mode(self, current_key):
        if current_key != 255:
            self.__last_background_key = current_key
        if current_key == ord('i'):
            self.__inverse_mask = not self.__inverse_mask

    def tune_color_threshold(self):
        size = (100, 600)
        color = [cv2.getTrackbarPos('H', self.__color_trackbar_image_window_name),
                 cv2.getTrackbarPos('S', self.__color_trackbar_image_window_name),
                 cv2.getTrackbarPos('V', self.__color_trackbar_image_window_name)]
        current_color = np.vstack(([np.full(size, color[0], dtype=np.uint8)],
                                   [np.full(size, color[1], dtype=np.uint8)],
                                   [np.full(size, color[2], dtype=np.uint8)])).transpose([1, 2, 0])
        current_color = cv2.cvtColor(current_color, cv2.COLOR_HSV2BGR)
        cv2.imshow(self.__color_trackbar_image_window_name, current_color)

    def find_color_mask(self, image, use_morphology=True):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.__get_thresholds()
        mask = cv2.inRange(hsv, self.__lower_threshold, self.__upper_threshold)
        if self.__lower_threshold_second is not None and self.__upper_threshold_second is not None:
            mask = mask + cv2.inRange(hsv, self.__lower_threshold_second, self.__upper_threshold_second)

        if use_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        return mask

    def get_background_mask(self, image_color_mask):
        return cv2.bitwise_not(image_color_mask)

    def get_final_image(self, image, background_mask, image_color_mask):
        if self.__last_background_key <= ord('0') or self.__last_background_key > ord('4'):
            return image

        if self.__inverse_mask:
            background_mask = cv2.bitwise_not(background_mask)
            image_color_mask = cv2.bitwise_not(image_color_mask)

        image_part = cv2.bitwise_and(image, image, mask=background_mask)
        if self.__last_background_key == ord('1'):
            background_part = cv2.bitwise_and(self.__white_noise, self.__white_noise, mask=image_color_mask)
        elif self.__last_background_key == ord('2'):
            background_part = cv2.bitwise_and(self.__camouflage1, self.__camouflage1, mask=image_color_mask)
        elif self.__last_background_key == ord('3'):
            background_part = cv2.bitwise_and(self.__camouflage2, self.__camouflage2, mask=image_color_mask)
        elif self.__last_background_key == ord('4'):
            background_part = cv2.bitwise_and(self.__background, self.__background, mask=image_color_mask)

        return cv2.addWeighted(image_part, 1, background_part, 1, 0)

    def __get_thresholds(self):
        h_mean = cv2.getTrackbarPos('H', self.__color_trackbar_image_window_name)
        s_mean = cv2.getTrackbarPos('S', self.__color_trackbar_image_window_name)
        v_mean = cv2.getTrackbarPos('V', self.__color_trackbar_image_window_name)
        h_radius = cv2.getTrackbarPos('H\nradius', self.__color_trackbar_image_window_name)
        s_radius = cv2.getTrackbarPos('S\nradius', self.__color_trackbar_image_window_name)
        v_radius = cv2.getTrackbarPos('V\nradius', self.__color_trackbar_image_window_name)

        s_lower = clamp(s_mean - s_radius, 0, 255)
        s_upper = clamp(s_mean + s_radius, 0, 255)
        v_lower = clamp(v_mean - v_radius, 0, 255)
        v_upper = clamp(v_mean + v_radius, 0, 255)

        if h_mean < h_radius or 180 - h_mean < h_radius:
            h_lower1, h_upper1, h_lower2, h_upper2 = split_range(h_mean, h_radius, 180)
            self.__lower_threshold = np.array([h_lower1, s_lower, v_lower])
            self.__upper_threshold = np.array([h_upper1, s_upper, v_upper])
            self.__lower_threshold_second = np.array([h_lower2, s_lower, v_lower])
            self.__upper_threshold_second = np.array([h_upper2, s_upper, v_upper])
        else:
            self.__lower_threshold = np.array([h_mean - h_radius, s_lower, v_lower])
            self.__upper_threshold = np.array([h_mean + h_radius, s_upper, v_upper])
            self.__lower_threshold_second = None
            self.__upper_threshold_second = None


def clamp(value, lower, upper):
    if value < lower:
        return lower
    elif value > upper:
        return upper
    return value


def split_range(value, radius, upper_threshold):
    lower_value = value - radius
    upper_value = value + radius
    down_looping = lower_value < 0
    return (0, upper_value, upper_threshold + lower_value, upper_threshold)\
        if down_looping else\
           (0, upper_value - upper_threshold, lower_value, upper_threshold)


def get_white_noise_image(shape, mean, cov):
    return np.reshape(list(map(np.uint8, np.random.uniform(mean, cov, reduce(lambda x, y: x * y, shape)))),
                      shape)


def nothing(x):
    pass
