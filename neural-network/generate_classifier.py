import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.optimizers import RMSprop
from glob import glob
import helper

def new_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(helper.NUM_CLASSES))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    return model


def cifar_model():
    model = Sequential()
    model.add(Conv2D(30, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(14, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(helper.NUM_CLASSES, activation='softmax'))

    opt = RMSprop(learning_rate=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(helper.NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def main():
    features, labels = helper.load_dataset()
    labels = helper.convert_labels(labels, to_numbers=True)

    num_classes = 12
    cat_targets = to_categorical(np.array(labels), num_classes)
    # print(cat_targets.shape)
    # print(cat_targets[:10])

    (x_train, y_train), (x_test, y_test) = helper.split_to_sets(features, cat_targets)
    # print(labels[:10])

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    # model = larger_model()
    # model.fit(x_train, y_train,
    #           batch_size=100,
    #           epochs=100,
    #           validation_data=(x_test, y_test),
    #           shuffle=True)
    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

    model = new_cifar_model()
    model.fit(x_train, y_train,
              batch_size=28,
              epochs=500,
              validation_data=(x_test, y_test),
              shuffle=True)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Large CNN Error: %.2f%%" % (100-test_acc*100))
    print("test-loss: ", test_loss)

    # #predictions = model.predict(test_images)

    model.save('model_2.h5')



if __name__ == "__main__":
    main()
