# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:42:47 2016

@author: DanielleT
"""

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

#LOAD DATA
print 'Loading Data....'
t = time.time()
#loads featurevectors
features1 = pd.read_csv('ML/Features_data/caffe_features_train.csv',sep=',', header=None, iterator=True,chunksize=1000)
features =  concat(features1, ignore_index=True)
#loads links photo to business_id
train_photos = pd.read_csv('../downloads/input/train_photo_to_biz_ids.csv',sep=',')
#loads business to labels 
biz_ids = pd.read_csv('../downloads/input/train.csv',sep=',')
#loads order of featurevectors
photo_order = pd.read_csv('ML/Features_data/photo_order_train.csv',sep=',',header=None)
X_TRAIN = pd.merge(biz_ids, train_photos, on='business_id')
print time.time()-t,
print 'seconds needed.'

#create file with businessIDs, and labels in binary form
Y_TRAIN = pd.concat([X_TRAIN['business_id'],X_TRAIN['labels'].str.get_dummies(sep=' ')], axis=1)
Y_TRAIN = Y_TRAIN.drop_duplicates()
XX = pd.merge(X_TRAIN, Y_TRAIN, on='business_id')
del(XX['labels'])

##Load the data
np.load('trainSet.npy')
np.load('verifSet.npy')

nonelabelsXX = XX[XX.business_id.isin([1627, 2661, 2941, 430])]
trainSet_XX = XX[XX.business_id.isin(trainSet)]
trainSetXX = trainSet_XX.append(nonelabelsXX)
valSetXX = XX[XX.business_id.isin(verifSet)]
del(trainSetXX['business_id'])

tt = time.time()
photo_order.rename(columns={0:'photo_id'},inplace=True)
all_data = pd.concat([photo_order,features], axis=1)
photo_idsXX = trainSetXX.loc[:,'photo_id']
photo_idsXX = photo_idsXX.map(lambda x: str(x) + 'm.jpg')
newXX = pd.concat([photo_idsXX,trainSetXX.loc[:,'0':]],axis=1)
newXX.rename(columns={'0':'00','1':'11','2':'22','3':'33','4':'44','5':'55','6':'66','7':'77','8':'88'},inplace=True)
alles = pd.merge(newXX, all_data, on='photo_id')
x_data = alles.iloc[:,-4096:]
y_data = alles.loc[:,'00':'88']
x_data.to_csv('x_data_PerImage.csv',index=True)
y_data.to_csv('y_data_PerImage.csv',index=True)


#load the x_data en y_data
x_dat = pd.read_csv('x_data_PerImage.csv',sep=',')
y_dat = pd.read_csv('y_data_PerImage.csv',sep=',')



#TRAIN SVM
print 'Training SVM....'   
ti = time.time()
S = OneVsRestClassifier(SVC(kernel='poly')).fit(x_data, y_data)
#score = S.score(x,y)
print time.time() - ti

#save classifier
import pickle
# now you can save it to a file
with open('svm.pkl', 'wb') as f:
    pickle.dump(S, f)

## and later you can load it
#with open('filename.pkl', 'rb') as f:
#    clf = pickle.load(f)



#TESTDATA
t = time.time()
#loads featurevectors
features1 = pd.read_csv('ML/Features_data/caffe_features_test.csv',sep=',', header=None, iterator=True,chunksize=1000)
features_test =  concat(features1, ignore_index=True)
#loads links photo to business_id
test_photos = pd.read_csv('../downloads/input/test_photo_to_biz.csv',sep=',')
#loads order of featurevectors
photo_order_test = pd.read_csv('ML/Features_data/photo_order_test.csv',sep=',',header=None)
print t-time.time()

S.predict(features_test)


