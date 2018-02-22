#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:11:04 2018

@author: aron
"""

import sys
import os
import shutil

import numpy as np
import h5py

from scipy.signal import stft
from scipy.io.wavfile import read as readWav
import matplotlib.pyplot as plot

absPath = (os.path.abspath('.'))

fs = 16000


maxSample = 0;

"Looking for the longest file"
for emotion in emotions.values():   
    for root, dirs, files in os.walk(absPath + '/dataToProcess' + '/' + emotion):
        for file in files:
            fileName = root + '/' + file
            freq, data = readWav(fileName)
            if (data.size > maxSample):
                maxSample=data.size
                print("New longest file length: ",maxSample, "  name:", file)
                
                
"Appending zeros"                
for emotion in emotions.values():   
    for root, dirs, files in os.walk(absPath + '/dataToProcess' + '/' + emotion):
        n = len(files)
        dataToSave = np.zeros((n,65,2246))
        i = 0
        for file in files:
            fileName = root + '/' + file
            freq, data = readWav(fileName)
            data = np.append(data, np.zeros(maxSample - data.size))
            f, t, Zxx = stft(data, fs, nperseg=128, window = 'cosine', noverlap = 0.5 * 128)
            dataToSave[i]= abs(Zxx)
            i = i+1
    if os.path.exists(absPath + '/hdf5_data.h5'):
        method = 'a'
    else:
        method = 'w'
    
    with h5py.File(absPath + '/hdf5_data.h5', method) as hdf:
        hdf.create_dataset(emotion, data=dataToSave)
            
            
                