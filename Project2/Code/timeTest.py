# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:10:14 2016

@author: Laurens
"""

import csv
import cv2
import time
import numpy as np
import os
from sklearn.cluster import KMeans

path = 'C:/Users/Laurens/Documents/TeamGreaterThanBrains/Project2/code/skintest/'
#files = os.listdir('C:/Users/Laurens/Documents/uni/Proj2/train/')
files = os.listdir(path)
df = np.array(['img','x','y'])

t = time.time()
for f in files:
    km = KMeans(n_clusters = 3)

    im = cv2.imread(path + f)
    
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
    #print km.cluster_centers_
    #print skinMask[skMask==1]
    #cv2.imwrite(sys.argv[2], skinMask) # Second image
    
    #skin = cv2.bitwise_and(im, im, mask = skinMask)
    #cv2.imwrite('C:/Users/Laurens/Documents/Teamgreaterthanbrains/Project2/Code/skintest/MASK' + f, skinMask)         # Final image
    #print np.reshape(np.concatenate(([f] *3,clcenters[:,1],clcenters[:,2])),(3,3)).T
    df = np.vstack((df,np.reshape(np.concatenate(([f] *3,clcenters[:,1],clcenters[:,2])),(3,3)).T))

print time.time() - t
t = time.time()
with open('timingTest.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(df)
print time.time() - t