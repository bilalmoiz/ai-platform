# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:08:05 2019

@author: firstname.lastname
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlflow
import mlflow.keras
import keras
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50

from keras import optimizers
from keras import metrics
from PIL import Image
import matplotlib.pyplot as plt 
import sys


#model parameters for easy tuning'
    

def build_compile_model(nodes):
    xray_model = Sequential()
    xray_model.add(Conv2D(32,(3,3), activation='relu',input_shape=(img_size,img_size,3)))
    xray_model.add(MaxPooling2D(pool_size=(2,2)))
    
    xray_model.add(Conv2D(32, (3, 3), activation="relu"))               
    xray_model.add(MaxPooling2D(pool_size=(2,2)))    
    
    xray_model.add(Conv2D(32, (3, 3), activation="relu"))               
    xray_model.add(MaxPooling2D(pool_size=(2,2)))   
                     
    xray_model.add(Flatten())
    
    xray_model.add(Dense(activation='relu',units=nodes))                      
    xray_model.add(Dense(activation='sigmoid',units=1))
                          
                          
    xray_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
                          
             
    xray_model.summary()    

    return xray_model

def gen_data():         

    train_path = "chest-xray-pneumonia/chest_xray/chest_xray/train/"
    test_path = "chest-xray-pneumonia/chest_xray/chest_xray/val/"
    val_path = "chest-xray-pneumonia/chest_xray/chest_xray/test/"
         
                        
    # generate the training data
    train_imagegen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.3, height_shift_range = 0.1, width_shift_range = 0.05, horizontal_flip=True)
    training_data = train_imagegen.flow_from_directory(train_path, target_size = (img_size,img_size), batch_size = batch_size, class_mode = 'binary')
    
    # generate the validation data
    val_imagegen = ImageDataGenerator(rescale = 1./255)
    val_data = val_imagegen.flow_from_directory(val_path, target_size = (img_size,img_size), batch_size = batch_size, class_mode = 'binary')
    
    # generate the test data
    test_imagegen = ImageDataGenerator(rescale = 1./255)
    test_data = test_imagegen.flow_from_directory(test_path, target_size = (img_size,img_size), batch_size = batch_size, class_mode = 'binary')
    
    return [training_data, val_data, test_data]
    
def graph_data(final_model, run_uuid):
    
    plt.plot(final_model.history['val_loss'])
    plt.plot(final_model.history['loss'])
    plt.legend(['Training','Test'])
    plt.xlabel('# of epoch')
    plt.ylabel('Loss')
    plt.savefig("graphs/graph"+ run_uuid, bbox_inches = "tight")
  #  plt.show()

if __name__ == "__main__":
    
    
    
    if len(sys.argv) < 2:
        print("Using defaults parameters only")
    else: 
        print("Using experimental parameters" )
        
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    img_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    epoch = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    steps_per_epoch = 163
        
    # build model and customize number of nodes
    xray_model = build_compile_model(nodes)
    
    data = gen_data()
    
    #train model with parameters
    trained_model = xray_model.fit_generator(data[0], steps_per_epoch = steps_per_epoch, epochs = epoch, validation_data = data[1], validation_steps = 624//batch_size)
    scores = xray_model.evaluate_generator(data[2], 12)
    

        
    with mlflow.start_run():
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)
        mlflow.keras.log_model(xray_model, "models")
        
        
        mlflow.log_param('Batch Size', batch_size)
        mlflow.log_param('Image size', img_size)
        mlflow.log_param('Epochs', epoch)
        mlflow.log_param('Number of Nodes in FC layer', nodes)
        
      
        mlflow.log_metric('Average Loss', trained_model.history['loss'][-1])
        mlflow.log_metric('Validation Loss', trained_model.history['val_loss'][-1])
        mlflow.log_metric('Accuracy', trained_model.history['acc'][-1])
        mlflow.log_metric('Validation Accuracy', trained_model.history['val_acc'][-1])
  
        graphs = graph_data(trained_model,run_uuid)
        mlflow.log_artifact("graphs/graph" + run_uuid + ".png")

        
    
    
    


