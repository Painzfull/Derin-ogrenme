from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np


(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255


datagen = ImageDataGenerator()
datagen.fit(X_train)

for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=4, seed=499):
    for i in range(0,4):
        pyplot.subplot(220 +1 +i)
        pyplot.imshow(X_batch[i])
    pyplot.show()
    break

datagen = ImageDataGenerator(rotation_range=359)
datagen.fit(X_train)

for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=4, seed=499):
    for i in range(0,4):
        pyplot.subplot(220 +1 +i)
        pyplot.imshow(X_batch[i])
    pyplot.show()
    break