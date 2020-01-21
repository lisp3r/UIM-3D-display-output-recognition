"""
Methods to find device on the picture
"""

import cv2

import helpers

class DeviceFindError(helpers.GeneralRecognitionFailure):
    """ Main exception raised from here """
    pass
class SeveralDevicesFound(DeviceFindError):
    pass
class NoDeviceFound(DeviceFindError):
    pass

class DeviceFinder(object):

    def __init__(self, blur_size=5, edge_th1=0, edge_th2=100, strip_border=0.03):
        self.__blur_size = blur_size
        self.__edge_th1 = edge_th1
        self.__edge_th2 = edge_th2
        self.__strip_border = strip_border

    def find(self, image):
        """ Return device images found and stripped from the image """
        prepared_image = helpers.prepare_edged(
            image,
            self.__blur_size,
            self.__edge_th1,
            self.__edge_th2
        )

        contours, hierarchy = cv2.findContours(prepared_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rect_ids = helpers.get_polygon_ids(contours, corner_cnt=4, epsilon=0.01)

        device_contours = []
        for rc_id in rect_ids:
            imm_squares = self.__get_immediate_square_childs(contours, hierarchy, rc_id)
            if len(imm_squares) >= 8:
                device_contours.append(rc_id)

        device_candidates_count = len(device_contours)
        if device_candidates_count == 1:
            return helpers.strip_contour(image, contours[device_contours[0]], border=self.__strip_border)
        elif device_candidates_count > 1:
            raise SeveralDevicesFound("{} device-like objects on the image".format(device_candidates_count))
        else:
            raise NoDeviceFound("No countours found with 8 or more squares inside")

    def __get_immediate_square_childs(self, contours, hierarchy, rc_id):
        """
        Return the first square childlrens for a contours
        """
        imm_childs = helpers.get_immediate_children(rc_id, hierarchy)
        imm_squares = []
        for square in helpers.get_square_ids(contours, epsilon=0.2):
            if square in imm_childs:
                imm_squares.append(square)
        return imm_squares
