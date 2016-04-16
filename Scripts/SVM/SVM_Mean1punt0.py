# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:24:16 2016

@author: DTump
"""
#Trains the classifier based on the mean of a business and the corresponding labels.
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import PIL 
from PIL import Image
import os
import math
from pandas import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
import time 
from sklearn import datasets

##load information
test_photos = pd.read_csv('../../downloads/input/test_photo_to_biz.csv',sep=',')
y = pd.read_csv('Features_data/labels_train_y.csv', header=None)
x = pd.read_csv('Features_data/feats_train_x.csv',header=None)
tr = pd.read_csv('../../downloads/input/train.csv')
bizzz = X_TRAIN.business_id.unique()

##delete the businesses without labels
bizzz = np.delete(bizzz, np.where(bizzz==1627))
bizzz = np.delete(bizzz, np.where(bizzz==2661))
bizzz = np.delete(bizzz, np.where(bizzz==2941))
bizzz = np.delete(bizzz, np.where(bizzz==430))
bizzz = pd.DataFrame(bizzz, columns={'business_id'})

##attach businessids
newXX = pd.concat([bizzz,x],axis=1)
newYY = pd.concat([bizzz,y],axis=1)

##Load the data
trainset = np.load('trainSet.npy')
verifset = np.load('verifSet.npy')

#the xdata, ydata en businessorder
traindatax = newXX[newXX.business_id.isin(trainset)]
verifdatax = newXX[newXX.business_id.isin(verifset)]
verifdatabiz = verifdatax.iloc[:,0]
traindatax = traindatax.iloc[:,1:]
verifdatax = verifdatax.iloc[:,1:]

##labels
traindatay = newYY[newYY.business_id.isin(trainset)]
verifdatay = newYY[newYY.business_id.isin(verifset)]
traindatay = traindatay.iloc[:,1:]
verifdatay = verifdatay.iloc[:,1:]

#train the svm
S = OneVsRestClassifier(SVC(kernel='poly', probability=True)).fit(traindatax, traindatay)
predictions = S.predict_proba(verifdatax)

## Attach businessorder to probabilities
predictions = pd.DataFrame(predictions)
verifdatabiz = pd.DataFrame(verifdatabiz)
predictions=predictions.reset_index()
verifdatabiz = verifdatabiz.reset_index()
pred = pd.concat([verifdatabiz,predictions], axis=1)
pred.to_csv('verifset_mean1punt0.csv',index=True)