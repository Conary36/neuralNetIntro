import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist  # pull in data from google

(train_images, train_labels ), (test_images, test_labels) = data.load_data() # structure data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0 # truncate pixel color scale to 1 (255/255)
test_images = test_images/255.0 # truncate pixel color scale to 1 (255/255)
# print(train_images[7]) # print in console the structured layout of data

# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()
"""
NOW LETS BUILD NEURAL NETWORK MODEL
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy", test_acc)



