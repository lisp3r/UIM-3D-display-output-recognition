import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from glob import glob
import random
import os


ROOT = "/home/lisp3r/image-recognition"
DATASET = "{}/datasets".format(ROOT)
RESULT = "{}/result".format(ROOT)
LABELS = "{}/labels.txt".format(ROOT)
BACKUP_DATASETS = "{}/English Typed Alphabets And Numbers/English/Fnt".format(ROOT)

NUM_CLASSES = 12

def split_to_sets(features, labels):
    index = int(len(labels)/4)

    features = features.reshape((features.shape[0], 28, 28, 1)).astype('float32')
    x_train = features[:index]
    x_test = features[len(x_train):]
    # x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
    # x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = labels[:index]
    y_test = labels[len(y_train):]

    return (x_train, y_train), (x_test, y_test)

def load_labels():
    with open(LABELS, "r") as f:
        labels_str = f.readline()
    return np.array([x for x in list(labels_str)])

def load_images(images_path, image=None):
    if image:
        return cv2.imread("{}/{}".format(images_path, image), cv2.IMREAD_GRAYSCALE)
    images = []
    for img in glob("{}/*".format(images_path)):
        images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    return images

def prepare_features(features):
    prepared = []
    for image in features:
        flatten_image = image.flatten()
        prepared.append(flatten_image)
    features = np.array([np.array(xi) for xi in prepared])
    return features

def load_dataset():
    images = load_images(RESULT)
    labels = load_labels()
    features = prepare_features(images)
    return features, labels

def convert_labels(labels, to_numbers=False):
    convert_map = yaml.load(open("{}/convert_map.yml".format(ROOT), "r"), Loader=yaml.Loader).get("map")

    result_labels = []
    if to_numbers:
        # convert to numbers
        for label in labels:
            for key in convert_map:
                values = convert_map[key]
                if type(values) != list:
                    values = [values]
                if str(label) in values:
                    result_labels.append(key)
    else:
        for label in labels:
            result_labels.append(convert_map.get(str(label)))
    return result_labels

def check_sanity(features, labels):
    lbs = []
    for x in range(5):
        rand_elem = random.randint(0, 12000)

        image = features[rand_elem]
        cv2.imwrite('sanity_{}.png'.format(x), image)
        label = labels[rand_elem]
        lbs.append(label)
    with open("sanity_labels.txt", "w") as w:
        for x in lbs:
            w.write(x)
