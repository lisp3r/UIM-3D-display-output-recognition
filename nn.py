""" Helper to prepare and recognize numbers with ready network """
import logging

from keras.models import load_model
# Workaround for this: https://github.com/keras-team/keras/issues/13353
import keras.backend.tensorflow_backend as tb

import numpy as np
import cv2

class NNRecognizer(object):
    def __init__(self, model_file_name, show=False):
        self.__th1 = 120
        self.__th2 = 255
        self.__target_size = (28, 28)
        self.__show = show

        self.__model = load_model(model_file_name)
        self.__logger = logging.getLogger("NNRecognizer")
        self.__convert_map = ['0','1','2','3','4','5','6','7','8','9','e','+']

    def recognize(self, number_image):
        self.__logger.debug("Prepare image")
        gray = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)
        ret, prepared_img = cv2.threshold(gray, self.__th1, self.__th2, cv2.THRESH_BINARY_INV)
        self.__logger.debug("Resize image")
        resized = cv2.resize(prepared_img, self.__target_size, interpolation=cv2.INTER_AREA)
        self.__logger.debug("Invert")
        inverted = cv2.bitwise_not(resized)
        if self.__show:
            cv2.imshow("Prepared number", inverted)
        self.__logger.debug("Flatten")
        flatten = inverted.flatten()
        nparrayed = np.array([np.array(xi) for xi in [flatten]])
        reshaped = nparrayed.reshape((nparrayed.shape[0], self.__target_size[0], self.__target_size[1], 1)).astype('float32')
        prepared = reshaped / 255
        self.__logger.debug("Predicting")
        tb._SYMBOLIC_SCOPE.value = True
        predictions = self.__model.predict(prepared)
        prediction = predictions[0]
        self.__logger.debug("Predicting result: {}".format(prediction))

        return self.__convert_map[int(np.argmax(prediction))]
