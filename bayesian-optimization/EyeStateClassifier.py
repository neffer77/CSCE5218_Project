import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
import cv2
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class EyeStateClassifier:
    def __init__(self, train_data_path, test_data_path, num_filters, kernel_size, pool_size, dense_size, learning_rate, batch_size, activation,pic_size=140):
    # Convert continuous variables to discrete as needed
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_size = dense_size
        self.batch_size = int(batch_size)
        print(self.batch_size)
        self.learning_rate = learning_rate
        self.activation = activation
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.pic_size = pic_size
        self.model = None
        self.Train_x, self.Train_y, self.X_test, self.y_test, self.X_val, self.y_val = [], [], [], [], [], []
        self.features = {'Closed':0, 'Open':1}

    def load_and_preprocess_data(self):
        size = []
        for folder in os.listdir(self.train_data_path) :
            if folder == 'Closed' or folder == 'Open':
                files = gb.glob(pathname= str( self.train_data_path + folder + '/*.jpg'))
                for file in files: 
                    image = plt.imread(file)
                    size.append(image.shape)
            else:
                break
        
        for folder in  os.listdir(self.train_data_path) : 
            if folder == 'Closed' or folder == 'Open':
                files = gb.glob(pathname= str( self.train_data_path + folder + '/*.jpg'))
                for file in files: 
                    image = cv2.imread(file)
                    image_array = cv2.resize(image,(self.pic_size,self.pic_size))
                    self.Train_x.append(list(image_array))
                    self.Train_y.append(self.features[folder])
            else:
                break



        print(f'we have {len(self.Train_x)} items in Train_x')
        print(f'we have {len(self.Train_y)} items in Train_y')




        
        for folder in  os.listdir(self.test_data_path) : 
            if folder == 'Closed' or folder == 'Open':
                files = gb.glob(pathname= str( self.test_data_path + folder + '/*.jpg'))
                for file in files: 
                    image = cv2.imread(file)
                    image_array = cv2.resize(image , (self.pic_size,self.pic_size))
                    self.X_test.append(list(image_array))
                    self.y_test.append(self.features[folder])
            else:
                break


        print(f'we have {len(self.X_test)} items in X_Test')
        print(f'we have {len(self.y_test)} items in Y_Test')

        self.Train_x, self.X_val, self.Train_y, self.y_val = train_test_split(self.Train_x, self.Train_y, train_size=0.8, shuffle=True, random_state=0)

        self.Train_x = np.array(self.Train_x)
        self.Train_y = np.array(self.Train_y)

        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)

        temp2 = list(zip(self.X_test, self.y_test))
        random.shuffle(temp2)
        self.X_test, self.y_test = zip(*temp2)
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

        
    def getcode(self,n) : 
        for one , two in self.features.items() : 
            if n == two : 
                return one



    def create_model(self):
        self.model = Sequential([
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation,input_shape=(self.pic_size,self.pic_size,3)),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        MaxPooling2D(self.pool_size,self.pool_size),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),    
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        MaxPooling2D(self.pool_size,self.pool_size),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        MaxPooling2D(self.pool_size,self.pool_size),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        Conv2D(self.num_filters,kernel_size=(self.kernel_size,self.kernel_size),activation=self.activation),
        Flatten() ,    
        Dense(self.dense_size,activation=self.activation) ,    
        Dense(self.dense_size,activation=self.activation) ,
        Dense(self.dense_size,activation=self.activation) ,
        Dense(self.dense_size,activation=self.activation) ,
        Dense(self.dense_size,activation=self.activation) ,
        Dense(1,activation='sigmoid') ,    
        ])

        optimizer= tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def train_model(self):
        data_generator = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.2, 
                             brightness_range=(0.1, 0.8))
        epochs = 20
        Model = self.model.fit(data_generator.flow(self.Train_x,self.Train_y, batch_size=self.batch_size), epochs=epochs,
                                        validation_data=(self.X_val,self.y_val), validation_steps=1, verbose=1)
    

    def evaluate_model(self):
        Loss,Accuracy = self.model.evaluate(self.X_test, self.y_test,batch_size=self.batch_size)

        print('Test Loss is {}'.format(Loss))
        print('Test Accuracy is {}'.format(Accuracy ))
        return Accuracy

    def predict(self, X):
        y_pred = self.model.predict(self.X_test)
        pred = [1 * (x[0]>=0.5) for x in y_pred]
        print('Prediction Shape is {}'.format(y_pred.shape))


        CM = confusion_matrix(self.y_test, pred)

        sns.heatmap(CM, center=True)
        plt.show()

        print('Confusion Matrix is\n', CM)


        print(classification_report(self.y_test, pred))
        print(accuracy_score(self.y_test, pred))

    def visualize_results(self,Model):
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

    def save_model(self, file_path):
        self.model.save('cnnCat.h5')

    def load_model_local(self, file_path):
        self.model = load_model(file_path)

# Usage

