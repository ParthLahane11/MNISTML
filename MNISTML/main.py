import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv('../Datasets/MNIST/train.csv')
test = pd.read_csv('../Datasets/MNIST/test.csv')

Y_train = train['label']
X_train = train.drop(labels = ['label'], axis= 1)

X_train = X_train/255.0
test = test/255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes= 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)

model = Sequential()
model.add(Conv2D(32, kernel_size= 3, activation = 'relu', input_shape= (28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, kernel_size= 3, activation = 'relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, kernel_size= 3, activation = 'relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, kernel_size= 3, activation = 'relu', input_shape= (28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, kernel_size= 3, activation = 'relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, kernel_size= 3, activation = 'relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size= 64, epochs = 30, validation_data= (X_val, Y_val), verbose= 2)

results = model.predict(test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)
submission.to_csv("MNIST_submission.csv", index = False)