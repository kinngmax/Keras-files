import numpy as np
import os
import sklearn
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Dropout, Flatten

# Data Preparation:

par_dir = "E:/Tensorflow/notMNIST/notMNIST_small"
path = os.listdir(par_dir)
image_list = []
label_list = []
label=0

for folder in path:
    images = os.listdir(par_dir + '/' + folder)
    for image in images:
        if(os.path.getsize(par_dir +'/'+ folder +'/'+ image) > 0):
            img = cv2.imread(par_dir +'/'+ folder +'/'+ image, 1)
            image_list.append(img)
            label_list.append(label)
        else:
            print('File ' + par_dir +'/'+ folder +'/'+ image + ' is empty')
    label += 1

print("Looping done")

image_array = np.empty([len(image_list),28, 28, 3])
for x in range(len(image_list)):
    image_array[x] = np.array(image_list[x])

image_array = image_array.astype(np.float32)

label_array = np.array(label_list)

one_hot = np.eye(10)[label_array]

image_data, one_hot = sklearn.utils.shuffle(image_array, one_hot)

print("Data ready. Bon Apetiet!")


image_train, label_train = image_data[0:12800], one_hot[0:12800]
image_test, label_test = image_data[12800:17920], one_hot[12800:17920]

def get_train_image(input):
    
    batch_images = image_train[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_train[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label

def get_test_image(input):

    batch_images = image_test[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_test[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label

# Model Building

model = Sequential()

model.add(Convolution2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding='same', input_shape=(28,28,3), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid', data_format="channels_last"))

model.add(Convolution2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding='same', input_shape=(14,14,32), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid', data_format="channels_last"))
'''
model.add(Convolution2D(128, kernel_size=(5, 5), strides=(1,1), activation='relu', padding='same', input_shape=(56,56,64), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid', data_format="channels_last"))
'''
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.7))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(image_train, label_train, batch_size=128, epochs=1, verbose=1)

score = model.evaluate(image_test, label_test, batch_size=128, verbose=0)
