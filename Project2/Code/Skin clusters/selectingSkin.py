# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:10:14 2016

@author: Laurens

This selects patches of colour within a certain range in the HSV colour space,
these patches are correlated with the skin of the driver, and in theory they should
give us three clusters, one for each hand(/arm) and one for the head. The position of the centre of these clusters
is then taken as a feature for that image.

"""

import csv
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

path = 'C:/Users/Laurens/Documents/uni/MLP/proj2/cutTrain/'
#files = os.listdir('C:/Users/Laurens/Documents/uni/Proj2/train/')
folders = os.listdir(path)
df = np.array(['img','x','y'])

fcount = 0
for fold in folders:
    files = os.listdir(path + fold)
    for f in files:
        fcount+=1
        km = KMeans(n_clusters = 3)
    
        im = cv2.imread(path + fold + '/' + f)
        
        #im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
        lower = np.array([0, 10, 15], dtype = "uint8")
        upper = np.array([50, 173, 240], dtype = "uint8")
        converted = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skinMask = np.array(skinMask)
        w,h = skinMask.shape
        vals = np.reshape(skinMask,(w*h))
        x = range(w) * h
        y = np.repeat(range(h),w)
        skinCoords = np.vstack((vals,x,y)).T
        km.fit(skinCoords)
        clcenters = km.cluster_centers_

        df = np.vstack((df,np.reshape(np.concatenate(([f] *3,clcenters[:,1],clcenters[:,2])),(3,3)).T))
        if fcount%100 == 0:
            print fcount

with open('skinTrainFeatures.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(df)