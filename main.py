#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network project
@author: aron, joci
"""

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plot
from scipy.signal import stft
from scipy.io.wavfile import read as readWav

fileName = '/home/kenderak/_repos/EmotionRecognition/dataset/wav/03a01Fa.wav'
freq, data = readWav(fileName)

plot.plot(data)
plot.title('Time domain')
plot.ylabel('Amplitude')
plot.xlabel('Time [sec]')
plot.grid()
plot.axis('tight')
plot.show()

fs = 16000


f, t, Zxx = stft(data, fs, nperseg=128, window = 'cosine', noverlap = 0.5 * 128)
plot.pcolormesh(t, f, np.angle(Zxx), vmin=0, vmax=np.max(np.angle(Zxx)))
plot.title('Spectrogram')
plot.ylabel('Frequency [Hz]')
plot.xlabel('Time [sec]')
plot.axis('tight')
plot.show()
t = np.transpose(np.arange(0.00, 1.9, 1/250))
plot.plot(t,abs(Zxx[1,1:476]))


