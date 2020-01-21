""" Get and strip numbers from the display """
import logging

import cv2

import helpers


class NumbersFinder(object):

    def __init__(self, blur_size=5, th1=120, th2=255, show=False):
        self.__blur_size = blur_size
        self.__th1 = th1
        self.__th2 = th2
        self.__show = show
        self.__area_filter_threshold = 0.008
        self.__strip_border = -0.09
        self.__logger = logging.getLogger("NumberFinder")

    def __prepare(self, image):
        """ Prepare display image to find and strip contours required """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.__blur_size, self.__blur_size), 0)
        ret, threshed = cv2.threshold(blur, self.__th1, self.__th2, cv2.THRESH_BINARY_INV)
        if self.__show:
            cv2.imshow("Prepared to find numbers", threshed)
        return threshed

    def find(self, display_img):
        """
        Find and return sorted number images on display image
        """

        display_height, display_width, _ = display_img.shape
        display_area = display_height * display_width
        area_threshold = display_area * self.__area_filter_threshold

        self.__logger.debug("Prepare image and get contours")
        prepared_image = self.__prepare(display_img)
        if self.__show:
             cv2.imshow("Prepared to find numbers", prepared_image)
        contours, _ = cv2.findContours(prepared_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__logger.debug("Find all big enough contours")
        big_contours = [ctr for ctr in contours if helpers.get_area(ctr) >= area_threshold]
        self.__logger.debug("%s contours found", len(big_contours))
        self.__logger.debug("Sort contours by X axis")
        sorted_contours = helpers.sorted_by_axis(big_contours)
        self.__logger.debug("Strip controus from prepared image")
        sorted_imgs = [helpers.strip_contour(display_img, c, border=self.__strip_border) for c in sorted_contours]
        return sorted_imgs
