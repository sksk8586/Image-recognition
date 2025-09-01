import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import cv2

datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models

(training_img, training_labels), (testing_img, testing_labels) = datasets.cifar10.load_data()

training_img= training_img/255
testing_img = testing_img/255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# for i in range(16):
#     plt.subplot(4,4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_img[i], cmap= plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
#
# plt.show()
# save resources nd time
training_img = training_img[:20000]
training_labels = training_labels[:20000]

testing_img = testing_img[:4000]
testing_labels = testing_labels[:4000]


# building neutral network 0/10 understanding
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])
#
# model.fit(training_img, training_labels, epochs = 10, validation_data=(testing_img, testing_labels))
#
#
# loss, accuracy = model.evaluate(testing_img, testing_labels)
#
# print(f"loss: {loss}")
# print(f"accuracy: {accuracy}")
#
# model.save("imageClassifier.keras")

model = models.load_model('imageClassifier.keras')

img = cv2.imread("deer.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"prediction is: {class_names[index]}")

plt.show()