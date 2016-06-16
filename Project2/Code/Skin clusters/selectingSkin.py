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
        #Set upper and lower bounds for the skin colours:
        lower = np.array([0, 10, 15], dtype = "uint8")
        upper = np.array([50, 173, 240], dtype = "uint8")
        #convert the image to HSV colour space
        converted = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #apply the range, and perform a series of actions to smooth out the blobs
        skinMask = cv2.inRange(converted, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skinMask = np.array(skinMask)
        #Prepare the skinMask for clustering
        w,h = skinMask.shape
        vals = np.reshape(skinMask,(w*h))
        x = range(w) * h
        y = np.repeat(range(h),w)
        skinCoords = np.vstack((vals,x,y)).T
        #Fit 3 clusters to the skinMask, and store the cluster centres as features
        km.fit(skinCoords)
        clcenters = km.cluster_centers_

        df = np.vstack((df,np.reshape(np.concatenate(([f] *3,clcenters[:,1],clcenters[:,2])),(3,3)).T))
        if fcount%100 == 0:
            print fcount

with open('skinTrainFeatures.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(df)