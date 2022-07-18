###################################-------------Artificial Neural Network 256-128-------------###################################

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
from tensorflow.keras import layers
from tensorflow.keras import initializers
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import BatchNormalization
from keras.utils import np_utils
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
                 new_array = cv2.resize(img_array,(100,100))  #resize all images to 100x100
                 training_data.append([new_array, class_num]) #add images to training_data
             except Exception as e: # in the interest of keeping the output clean
                 pass


create_training_data()



X_train = []
y_train = []

#for loop to fill the X array and y list
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

#reshape X_train array to be a 3 column with 3 dimensions to apply it to neural net
X_train = np.array(X_train).reshape(-1,30000)


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

#reshape X_test array to be a 3 column with 3 dimensions to apply it to neural net
X_test = np.array(X_test).reshape(-1,30000)



#Normalize data to 0 and 1
X_train = X_train/255
X_test =  X_test/255

num_classes = 10
#one hot encoding for classes
y_train = np_utils.to_categorical(y_train,num_classes=num_classes)
y_test = np_utils.to_categorical(y_test,num_classes=num_classes)



########---------ANN Creation---------########

model = Sequential()
# 1st hidden layer
model.add(Dense(256,input_shape=(30000,)))
model.add(Activation('sigmoid'))
# 2nd hidden layer
model.add(Dense(128))
model.add(Activation('sigmoid'))
#Output layer
model.add(Dense(10))
model.add(Activation('softmax'))

#summary of the ANN
model.summary()


opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD")
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy',Recall(),Precision(),PrecisionAtRecall(0.5),F1Score(num_classes=10)])
history = model.fit(X_train,y_train,batch_size=128,epochs=80,validation_data=(X_test,y_test),verbose=1,shuffle=True)


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
plt.legend(['train', 'test'], loc='upper right')

#Plot Loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

#Plot Recall
plt.subplot(2,1,2)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

#Plot Precision
plt.subplot(2,1,2)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


#Plot PrecisionAtRecall
plt.subplot(2,1,2)
plt.plot(history.history['precision_at_recall'])
plt.plot(history.history['val_precision_at_recall'])
plt.title('model PrecisionAtRecall')
plt.ylabel('Precision_At_Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
classes = [0,1,2,3,4,5,6,7,8,9]

####--------Confusion Matrix--------####
conf_mat = confusion_matrix(np.argmax(y_test, axis=-1), y_pred)
f,ax=plt.subplots(figsize=(5,5))
# Normalize the confusion matrix.
conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
plt.title("Confusion matrix")
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap=plt.cm.Blues,linecolor="gray",fmt=".1f",ax=ax)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes, rotation=45)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
