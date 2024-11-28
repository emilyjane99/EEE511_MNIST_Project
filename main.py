import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D,MaxPool2D, UpSampling2D,Dropout

from keras._tf_keras.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#CODE BLOCK 1: SELECT 3 DIGITS
train_filter_digits = np.where((y_train == 2) | (y_train == 3) | (y_train == 6))
test_filter_digits = np.where((y_test == 2) | (y_test == 3) | (y_test == 6))

x_train = x_train[train_filter_digits]
y_train = y_train[train_filter_digits]
x_test = x_test[test_filter_digits]
y_test = y_test[test_filter_digits]

# to get the shape of the data
print("x_train shape:",x_train.shape)
print("x_test shape", x_test.shape)

plt.figure(figsize = (8,8))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.title(str(y_train[i]),fontsize = 16, color = 'purple', pad = 2)
  plt.imshow(x_train[i], cmap = plt.cm.binary )
  plt.xticks([])
  plt.yticks([])

plt.show()

#CODE BLOCK 2: Size/Selection of Validation and Test Datasets
val_images = x_test[:2700]
test_images = x_test[2700:]

print("initial test images shape")
print(test_images.shape)

#NORMALIZE AND RESHAPE
val_images = val_images.astype('float32') / 255.0
val_images = np.reshape(val_images,(val_images.shape[0],28,28,1))

test_images = test_images.astype('float32') / 255.0
test_images = np.reshape(test_images,(test_images.shape[0],28,28,1))

train_images = x_train.astype("float32") / 255.0
train_images = np.reshape(train_images, (train_images.shape[0],28,28,1))

#ADD NOISE
factor = 0.39
train_noisy_images = train_images + factor * np.random.normal(loc = 0.0,scale = 1.0,size = train_images.shape)
val_noisy_images = val_images + factor * np.random.normal(loc = 0.0,scale = 1.0,size = val_images.shape)
test_noisy_images = test_images + factor * np.random.normal(loc = 0.0,scale = 1.0,size = test_images.shape)

# here maximum pixel value for our images may exceed 1 so we have to clip the images
train_noisy_images = np.clip(train_noisy_images,0.,1.)
val_noisy_images = np.clip(val_noisy_images,0.,1.)
test_noisy_images = np.clip(test_noisy_images,0.,1.)

#SHOW IMAGES AFTER NOISE
plt.figure(figsize = (8,8))

for i in range(25):
      plt.subplot(5,5,i+1)
      plt.title(str(y_train[i]),fontsize = 16, color = 'blue', pad = 2)
      plt.imshow(train_noisy_images[i].reshape(1,28,28)[0], cmap = plt.cm.binary )
      plt.xticks([])
      plt.yticks([])

plt.show()

#AUTOENCODER MODEL
model = Sequential()
# encoder network
model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = 'relu', padding = 'same', input_shape = (28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (2,2),strides = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (2,2),strides = (2,2), activation = 'relu', padding = 'same'))

# decoder network
model.add(Conv2D(filters = 256, kernel_size = (2,2), activation = 'relu', padding = 'same'))

model.add(tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2),activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = 'relu', padding = 'same'))


model.add(tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (2,2),strides = (2,2), activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 32, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(filters = 1, kernel_size = (2,2), activation = 'relu', padding = 'same'))

# to get the summary of the model
model.summary()

#COMPILE MODEL
OPTIMIZER =  tf.keras.optimizers.Adam(learning_rate = 0.010)
LOSS = 'mean_squared_error'
model.compile(optimizer =OPTIMIZER, loss = LOSS, metrics = ['accuracy'])

#FIT MODEL
EPOCHS = 5
BATCH_SIZE = 128
VALIDATION = (val_noisy_images, val_images)
history = model.fit(train_noisy_images, train_images,batch_size = BATCH_SIZE,epochs = EPOCHS, validation_data = VALIDATION)

#EVALUATE MODEL
plt.subplot(2,1,1)
plt.plot( history.history['loss'], label = 'loss')
plt.plot( history.history['val_loss'], label = 'val_loss')
plt.legend(loc = 'best')
plt.subplot(2,1,2)
plt.plot( history.history['accuracy'], label = 'accuracy')
plt.plot( history.history['val_accuracy'], label = 'val_accuracy')
plt.legend(loc = 'best')
plt.show()

#DENOISE USING AUTOENCODER
plt.figure(figsize=(18, 18))
print(test_images.shape)

for i in range(10, 19):
    plt.subplot(9, 9, i)
    if i == 14:
        plt.title('Real Images', fontsize=25, color='Green')
    plt.imshow(test_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.show()

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    if (i == 15):
        plt.title('Noised Images', fontsize=25, color='red')
    plt.subplot(9, 9, i)
    plt.imshow(test_noisy_images[i].reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.show()

plt.figure(figsize=(18, 18))
for i in range(10, 19):
    if (i == 15):
        plt.title('Denoised Images', fontsize=25, color='Blue')

    plt.subplot(9, 9, i)
    plt.imshow(model.predict(test_noisy_images[i].reshape(1, 28, 28, 1)).reshape(1, 28, 28)[0], cmap=plt.cm.binary)
plt.show()

