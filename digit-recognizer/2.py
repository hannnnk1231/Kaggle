import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.Session(config=config)

def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

Y_train=train['label']
X_train=train.drop(labels=['label'],axis=1)
test=test/255.0
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
Y_train=to_categorical(Y_train, num_classes=10)
X_train, X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 0.1,
                                   zoom_range = 0.25,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255.)

model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), padding = 'same', input_shape = (28,28,1)))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(64, (3,3), padding = 'same'))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3,3), padding = 'same'))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(128, (3,3), padding = 'same'))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3,3), padding = 'same'))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(256, (3,3), padding = 'same'))
model.add(layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation = 'softmax'))

initial_learningrate=2e-3
batch_size = 1024
epochs = 50
input_shape = (28, 28, 1)

model.compile(optimizer = RMSprop(lr=initial_learningrate) , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(train_datagen.flow(X_train,Y_train,batch_size=batch_size),
                              steps_per_epoch=100,
                              epochs=epochs,
                              callbacks=[LearningRateScheduler(lr_decay)],
                              validation_data=test_datagen.flow(X_val,Y_val),
                              validation_steps=50)

results=model.predict(test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen_new.csv",index=False)