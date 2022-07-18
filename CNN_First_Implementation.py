###################################------------- Convolutional Neural Network -------------###################################

#import libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import Precision
from keras.metrics import Recall
from keras.metrics import PrecisionAtRecall
from tensorflow_addons.metrics import F1Score
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#mount google drive
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/MyDrive"

DATADIR = "/content/drive/MyDrive/My_Dataset/Train_Dataset"
Categories = ["0","1","2","3","4","5","6","7","8","9"]


training_data = []

#then we want to create a function for training_data to be implemented to
def create_training_data():
     for category in Categories:
         path = os.path.join(DATADIR,category)  #create path to Dataset
         class_num = Categories.index(category) #create a variable class_num which will display the label of each class (0,1,..,9 )
         for img in os.listdir(path):  #iterate over each image per each sign
             try:
                 img_array = cv2.imread(os.path.join(path,img)) #convert to img_array
                 new_array = cv2.resize(img_array,(100,100))  #resize all images to 50x50
                 training_data.append([new_array, class_num]) #add images to training_data
             except Exception as e: # in the interest of keeping the output clean
                 pass
             #except OSError as e:
             #   print("OSError,Bad img most likely",e, os.path.join(path,img))
             #except Exception as e:
             #   print("general exception", e, os.path.join(path,img))

create_training_data()



X_train = []
y_train = []

#for loop to fill the X array and y list
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

#reshape X_train array to be a 3 column with 3 dimensions to apply it to neural net
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],3)



DATADIR2 = "/content/drive/MyDrive/My_Dataset/Test_Dataset"

#now we want to create an array for training_data
testing_data = []

#then we want to create a function for training_data to be implemented to
def create_testing_data():
     for category in Categories:
         path = os.path.join(DATADIR2,category)  #create path to Dataset
         class_num = Categories.index(category) #create a variable class_num which will display the label of each class (0,1,..,9 )
         for img in os.listdir(path):  #iterate over each image per each sign
             try:
                 img_array = cv2.imread(os.path.join(path,img)) #convert to img_array
                 new_array = cv2.resize(img_array,(100,100))  #resize all images to 100x100
                 testing_data.append([new_array, class_num]) #add images to training_data
             except Exception as e: # in the interest of keeping the output clean
                 pass

create_testing_data()



#feature and labels set
X_test = []
y_test = []

#for loop to fill the X array and y list
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)

#reshape X_test array
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],3)



#one hot encoding for classes
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes=num_classes)
y_test = np_utils.to_categorical(y_test,num_classes=num_classes)



###########---------- CNN Creation ----------###########

model = Sequential()

#1st hidden layer
model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd hidden layer
model.add(Conv2D(32,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#Fully Connected Layer
model.add(Dense(128))

#2nd Fully Connected Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,batch_size=128,validation_data=(X_test,y_test), epochs=10,verbose=1)

#########-------Test Set Implementation-------#########
score = model.evaluate(X_test,y_test)
print('Test loss: ',score[0])
print('Test accuracy: ',score[1])


#Plot Accuracy
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


#Plot Loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
