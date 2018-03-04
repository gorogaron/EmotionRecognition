#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:23:19 2018

@author: aron, jozsef
"""
import sys
import os
import shutil

"print(os.path.abspath('createFolders.py'))"



if not os.path.exists('dataToProcess'):
    os.mkdir('dataToProcess')

emotions = {  'W' : 'anger', 
              'L' : 'boredom', 
              'E' : 'disgust',
              'A' : 'fear',
              'F' : 'happiness',
              'T' : 'sadness',
              'N' : 'neutral'  }

for key, value in emotions.items():
    name = os.path.abspath('.')+ '/dataToProcess/' + value
    if not os.path.exists(name):
        os.mkdir(name)
        print("Created dir : " + value)

print("Directories have been created.")
print("Copying files...")


for root, dirs, files in os.walk(os.path.abspath('.') + '/dataset/wav'):
    for file in files:
        key = file[5]
        srcFile = root + '/' + file
        dstFile = os.path.abspath('.') + '/dataToProcess' + '/' + emotions[key] + '/' + file        
        shutil.copy(srcFile, dstFile)
        print("Copying " + file)

print("Done")
        
        
        
        
        
        
        