#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob as gb
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
import os


train_data = '/dataset_new/train/'
test_data = '/dataset_new/test/'


for folder in  os.listdir(train_data): 
    files = gb.glob(pathname= str( train_data + folder + '/*.jpg'))
    print(f'For data , found {len(files)} in folder {folder}')


features = {'Closed':0, 'Open':1}
def getcode(n) : 
    for one , two in features.items() : 
        if n == two : 
            return one


size = []
for folder in os.listdir(train_data) :
    if folder == 'Closed' or folder == 'Open':
        files = gb.glob(pathname= str( train_data + folder + '/*.jpg'))
        for file in files: 
            image = plt.imread(file)
            size.append(image.shape)
    else:
        break
pd.Series(size).value_counts


pic_size = 140
Train_x = []
Train_y = []
for folder in  os.listdir(train_data) : 
    if folder == 'Closed' or folder == 'Open':
        files = gb.glob(pathname= str( train_data + folder + '/*.jpg'))
        for file in files: 
            image = cv2.imread(file)
            image_array = cv2.resize(image,(pic_size,pic_size))
            Train_x.append(list(image_array))
            Train_y.append(features[folder])
    else:
        break


print(f'we have {len(Train_x)} items in Train_x')
print(f'we have {len(Train_y)} items in Train_y')


X_test = []
y_test = []
for folder in  os.listdir(test_data) : 
    if folder == 'Closed' or folder == 'Open':
        files = gb.glob(pathname= str( test_data + folder + '/*.jpg'))
        for file in files: 
            image = cv2.imread(file)
            image_array = cv2.resize(image , (pic_size,pic_size))
            X_test.append(list(image_array))
            y_test.append(features[folder])
    else:
        break


print(f'we have {len(X_test)} items in X_Test')
print(f'we have {len(y_test)} items in Y_Test')


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(Train_x),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(Train_x[i])   
    plt.axis('off')
    plt.title(getcode(Train_y[i]))


Train_x, X_val, Train_y, y_val = train_test_split(Train_x, Train_y, train_size=0.8, shuffle=True, random_state=0)

Train_x = np.array(Train_x)
Train_y = np.array(Train_y)

X_val = np.array(X_val)
y_val = np.array(y_val)

temp2 = list(zip(X_test, y_test))
random.shuffle(temp2)
X_test, y_test = zip(*temp2)
X_test, y_test = np.array(X_test), np.array(y_test)


model = Sequential([
        Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(pic_size,pic_size,3)),
        Conv2D(32,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,kernel_size=(5,5),activation='relu'),    
        Conv2D(64,kernel_size=(5,5),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,kernel_size=(5,5),activation='relu'),
        Conv2D(128,kernel_size=(5,5),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256,kernel_size=(3,3),activation='relu'),
        Conv2D(256,kernel_size=(3,3),activation='relu'),
        Flatten() ,    
        Dense(512,activation='relu') ,    
        Dense(512,activation='relu') ,
        Dense(256,activation='relu') ,
        Dense(256,activation='relu') ,
        Dense(128,activation='relu') ,
        Dense(1,activation='sigmoid') ,    
        ])


optimizer= tf.keras.optimizers.Adam(learning_rate=0.00008)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


data_generator = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.2, 
                             brightness_range=(0.1, 0.8))


epochs = 20
Model = model.fit_generator(data_generator.flow(Train_x,Train_y, batch_size=32), epochs=epochs,
                                     validation_data=(X_val,y_val), validation_steps=1, verbose=1)


Loss,Accuracy = model.evaluate(X_test, y_test,batch_size=32)

print('Test Loss is {}'.format(Loss))
print('Test Accuracy is {}'.format(Accuracy ))



plt.plot(Model.history['accuracy'])
plt.plot(Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# # Predict testing data

y_pred = model.predict(X_test)
pred = [1 * (x[0]>=0.5) for x in y_pred]
print('Prediction Shape is {}'.format(y_pred.shape))


CM = confusion_matrix(y_test, pred)

sns.heatmap(CM, center=True)
plt.show()

print('Confusion Matrix is\n', CM)


print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(pred[i]))


model.save('cnnCat.h5')


"""
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('driver_state.h5')

test_image = cv2.imread('test_image.jpg') 
preprocessed_image = preprocess_image(test_image) 

input_image = np.expand_dims(preprocessed_image, axis=0)

segmentation_mask = model.predict(input_image)

segmented_image = np.argmax(segmentation_mask, axis=-1)[0]

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




