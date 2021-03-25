from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
from keras import optimizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import os
import datetime
import tensorflow as tf
import numpy as np 

#Parameters
train_data_path = '224X224/coba2/train'
test_data_path = '224X224/coba2/test'
WIDTH = 224
HEIGHT = 224
num_of_train_samples = 15984
num_of_test_samples = 3996
batch_size = 32

INPUT_SHAPE = (WIDTH, HEIGHT, 3)   #change to (SIZE, SIZE, 3)
model = Sequential()
model.add(Conv2D(16, (3, 3), padding ='same', input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding ='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding ='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding ='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

opt=Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])

#Tensorboard log
log_dir = '224x224/coba2/tf-log/tf-log(epoch=5, lr=0.001, Op=adam)/' 
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0) 
cbks = [tb_cb]

print(model.summary())

train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_path,  # this is the input directory
        target_size=(224, 224),  # all images will be resized to 64x64
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = validation_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

#We can now use these generators to train our model. 
history = model.fit_generator(
            train_generator,
            steps_per_epoch=1000 // batch_size,    #The 2 slashes division return rounded integer
            epochs=5,
            validation_data=validation_generator,
            validation_steps=100 // batch_size,
            callbacks=cbks)

os.mkdir('224x224/coba2/models/model(epoch=5, lr=0.001, Op=adam)') 
model.save('224x224/coba2/models/model(epoch=5, lr=0.001, Op=adam)/model.h5')
model.save_weights('224x224/coba2/models/model(epoch=5, lr=0.001, Op=adam)/weights.h5')

#Graphic
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.figure()
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('Accuracy_224x224_coba2_lr=0.001_epoch=5.jpg')

#Train and validation loss
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('Loss_224x224_coba2_lr=0.001_epoch=5.jpg')
