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

print(train_images[7]) # print in console the structured layout of data

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
