""" Find a display on device image"""

import cv2
import helpers

class NoDisplayFound(helpers.GeneralRecognitionFailure):
    pass

class DisplayFinder(object):

    def __init__(self, blur_size=3, edge_th1=20, edge_th2=50):
        self.__blur_size = blur_size
        self.__edge_th1 = edge_th1
        self.__edge_th2 = edge_th2

    def find(self, image):
        """
        Find display on a device image.
        Display is a rectangle contour with greates width + height
        """

        device_height, device_width, _ = image.shape
        device_area = device_height * device_width

        prepared_image = helpers.prepare_edged(
            image, self.__blur_size, self.__edge_th1, self.__edge_th2
        )

        contours, _ = cv2.findContours(prepared_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rect_ids = helpers.get_polygon_ids(contours, 4)

        rectangle_areas = [ helpers.get_area(contours[id]) for id in rect_ids ]

        if max(rectangle_areas) < 0.3 * device_area:
            raise NoDisplayFound("Unable to find display on device image provided")

        screen_contour_id = rect_ids[rectangle_areas.index(max(rectangle_areas))]

        return helpers.strip_contour(image, contours[screen_contour_id])
