import os
import shutil
from random import randrange
from glob import glob
import cv2
import subprocess
import helper


ROOT = "/home/lisp3r/image-recognition"
DATASET = "{}/datasets".format(ROOT)
RESULT = "{}/result".format(ROOT)
LABELS = "{}/labels.txt".format(ROOT)
BACKUP_DATASETS = "{}/English Typed Alphabets And Numbers/English/Fnt".format(ROOT)


def check_sanity():

    print("Sanity check")

    sanity_test_path = "{}/sanity_test".format(ROOT)
    if os.path.isdir(sanity_test_path):
        shutil.rmtree(sanity_test_path)
    os.mkdir(sanity_test_path)

    labels = helper.load_labels()

    test_labels = []
    for i in range(5):
        rand_elem = randrange(12000)
        print("Rand item:", rand_elem)
        print("Fit label:", labels[rand_elem])
        test_img = helper.load_images(RESULT, "{}.png".format(rand_elem))
        cv2.imwrite("{}/sanity_{}.png".format(sanity_test_path, i), test_img)
        test_labels.append(labels[rand_elem])
    with open("{}/sanity_labels.txt".format(sanity_test_path), "w") as w:
        for x in test_labels:
            w.write(x)

def restore_datasets():
    os.mkdir(DATASET)
    for folder in glob("{}/*".format(BACKUP_DATASETS)):
        shutil.copytree(folder, "{}/{}".format(DATASET, os.path.basename(folder)))

def update_lables():
    result_lables = []
    for i in os.listdir(DATASET):
        result_lables.append(i)
    return result_lables


def main():
    if os.path.isdir(RESULT):
        shutil.rmtree(RESULT)

    if os.path.isdir(DATASET):
        shutil.rmtree(DATASET)

    os.mkdir(RESULT)
    restore_datasets()
    lables = update_lables()
    result_lables = []

    ind = 0

    while len(os.listdir(DATASET)):
        print("Lables:", lables)

        dir_index = randrange(len(lables))
        cur_dir_name = lables[dir_index]

        cur_dir_path = os.path.join(DATASET, cur_dir_name)
        print("Work dir: {}".format(cur_dir_path))

        try:
            cur_file_name = os.listdir(cur_dir_path)[0]
            cur_file = os.path.join(cur_dir_path, cur_file_name)

            print("Work file: {}".format(cur_file_name))
            shutil.copy(cur_file,
                        os.path.join(RESULT, "{}.png".format(ind)))
            os.remove(cur_file)
            result_lables.append(cur_dir_name)
            ind = ind+1
        except:
            print("Directory is empty or not exisis")
            try:
                shutil.rmtree(cur_dir_path)
            except:
                pass

    with open(LABELS, "w+") as f:
        for i in result_lables:
            f.write(i)

    # resize to 28x28
    # for img in glob("{}/*".format(RESULT)):
    #     print("Resizing {}".format(os.path.basename(img)))
    #     cv2.resize(img, (28, 28))



    for img in glob("{}/*".format(RESULT)):
        print("Resizing {}".format(os.path.basename(img)))
        subprocess.check_output(["convert", img, "-resize", "28x28!", img])

    check_sanity()

if __name__ == "__main__":
    main()
