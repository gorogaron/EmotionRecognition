#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:11:04 2018

@author: aron, jozsef
"""

import os

import numpy as np
import h5py

from scipy.signal import stft
from scipy.io.wavfile import read as readWav
import matplotlib.pyplot as plot
import math
absPath = (os.path.abspath('.'))

fs = 16000




"Looking for the longest file"
"""maxSample = 0;
for emotion in emotions.values():   
    for root, dirs, files in os.walk(absPath + '/dataToProcess' + '/' + emotion):
        for file in files:
            fileName = root + '/' + file
            freq, data = readWav(fileName)
            if (data.size > maxSample):
                maxSample=data.size
                print("New longest file length: ",maxSample, "  name:", file)"""
                
                
"Appending zeros and STFT"                
"""for emotion in emotions.values():   
    for root, dirs, files in os.walk(absPath + '/dataToProcess' + '/' + emotion):
        n = len(files)
        dataToSave = np.zeros((n,128,1124))
        i = 0
        for file in files:
            fileName = root + '/' + file
            freq, data = readWav(fileName)
            data = np.append(data, np.zeros(maxSample - data.size))
            f, t, Zxx = stft(data, fs, nperseg=255, window = 'cosine', noverlap = 0.5 * 255)
            dataToSave[i]= abs(Zxx) 
            i = i+1
           """

for emotion in emotions.values():
    for root, dirs, files in os.walk(absPath + '/dataToProcess' + '/' + emotion):
        dataToSave = []
        isEmpty = True
        for file in files:            
            fileName = root + '/' + file
            freq, data = readWav(fileName) 
            f, t, Zxx = stft(data, fs, nperseg=255, window = 'cosine', noverlap = 0.5 * 255)
            n = math.floor(Zxx.shape[1]/Zxx.shape[0]);
            for i in range(0, n):
                if isEmpty:
                    nCols = 0
                    isEmpty = False
                else:
                    nCols = dataToSave.shape[1]
                dataToSave = np.append(dataToSave, abs(Zxx[:,i*128:(i+1)*128])).reshape(128,nCols+128)
    
    if os.path.exists(absPath + '/hdf5_data.h5'):
        method = 'a'
    else:
        method = 'w'
    
    with h5py.File(absPath + '/hdf5_data.h5', method) as hdf:
        hdf.create_dataset(emotion, data=dataToSave)
       
