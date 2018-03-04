#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:40:27 2018

@author: aron, jozsef
"""
import os
import h5py
import math
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from sklearn import preprocessing
from keras import utils

tfBackEnd = True
if (tfBackEnd):
	import tensorflow as tf
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	from keras import backend as K
	K.set_session(sess)

def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
       shuffled_a[new_index] = a[old_index]
       shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def loadDataForCNN():

    absPath = (os.path.abspath('.'))
    dataset = []
    labels = []
    
    with h5py.File(absPath + '/hdf5_data.h5', 'r') as hdf:
        ls = list(hdf.keys())
        for i in range(0, 7):
            data = np.array(hdf.get(ls[i]))
            labels = np.append(labels, i*np.ones((1,math.floor(data.shape[1]/128))))
            dataset = np.append(dataset, np.array(data))
    
    dataset = np.reshape(dataset,(1180,128,128))    #1180, mert ennyi 128x128as képünk van összesen
    
    scaler = preprocessing.StandardScaler()
    
    for oneData in dataset:
        scaler.partial_fit(oneData.reshape(-1,1))
    
    scaledData = (dataset - scaler.mean_)/scaler.scale_    
    labels, scaledData = shuffle_in_unison(labels, scaledData)
    
    trainSize = 0.7
    validationSize = 0.15
    testSize = 1 - trainSize - validationSize
    
    trainX = scaledData[0:math.floor(trainSize*scaledData.shape[0])]
    testX = scaledData[math.floor(trainSize*scaledData.shape[0]):math.floor((testSize+trainSize)*scaledData.shape[0])]    
    validX = scaledData[math.floor((testSize+trainSize)*scaledData.shape[0]):scaledData.shape[0]]        
    
    trainY = labels[0:math.floor(trainSize*labels.shape[0])]
    testY = labels[math.floor(trainSize*labels.shape[0]):math.floor((testSize+trainSize)*labels.shape[0])]    
    validY = labels[math.floor((testSize+trainSize)*labels.shape[0]):labels.shape[0]] 
    
    trainX = trainX.reshape((trainX.shape[0],trainX.shape[1], trainX.shape[2], 1))
    testX = testX.reshape((testX.shape[0],testX.shape[1], testX.shape[2], 1))
    validX = validX.reshape((validX.shape[0],validX.shape[1], validX.shape[2], 1))
    
    trainY = trainY.reshape((trainY.shape[0],1))
    testY = testY.reshape((testY.shape[0],1))
    validY = validY.reshape((validY.shape[0],1))
    
    trainY = utils.np_utils.to_categorical(trainY)
    testY = utils.np_utils.to_categorical(testY)
    validY = utils.np_utils.to_categorical(validY)
    
    return trainX, testX, validX, trainY, testY, validY

def createCNN(shape):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(7, activation = 'sigmoid'))
    
    #model.summary()
    
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.8, decay = 1e-6)
    model.compile(optimizer = sgd, loss='mean_squared_error', metrics=['acc'])
    
    return model

def createCNN_LSTM():
    model = models.Sequential()
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3,3), activation = 'relu'), input_shape = (4, 128, 128, 1)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    
    model.add(layers.TimeDistributed(layers.Flatten()))
    
    LSTMunits = 32
    model.add(layers.LSTM(units = LSTMunits, activation = 'relu', return_sequences = 'true'))
    model.add(layers.LSTM(units = LSTMunits, activation = 'relu', return_sequences = 'true'))
    
    model.add(layers.Reshape((LSTMunits*4,), input_shape=(4,LSTMunits)))
    
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(7, activation = 'sigmoid'))
    
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.8, decay = 1e-6)
    model.compile(optimizer = sgd, loss='mean_squared_error', metrics=['acc'])
    
    return model

def plotResults(history):
    import matplotlib.pyplot as plt
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    
    trainX, testX, validX, trainY, testY, validY = loadDataForCNN()
    
    model = createCNN(trainX.shape[1:])
    
    history = model.fit(trainX, trainY, epochs = 25, batch_size = 20 , validation_data = (validX, validY))
    
    plotResults(history)
    