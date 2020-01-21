import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import helper
import subprocess

model = load_model("saved_models/keras_cifar10_trained_model.h5")
images_names = ["0.png", "12199.png"]

images = []
for img in images_names:
    images.append(cv2.imread(img))


# prepared_images = helper.prepare_features(images)

prepared_images = np.array([np.array(xi) for xi in images])
print(prepared_images.shape)
prepared_images = prepared_images.reshape((prepared_images.shape[0], 32, 32, 3)).astype('float32')
prepared_images = prepared_images / 255

predictions = model.predict(prepared_images)

for x in predictions:
    print("All data:")
    print(x)
    print("Predict:", int(np.argmax(x)), helper.convert_labels([int(np.argmax(x))]))

plt.figure(figsize=(5, 5))
plt.imshow(images[0])
plt.show()
