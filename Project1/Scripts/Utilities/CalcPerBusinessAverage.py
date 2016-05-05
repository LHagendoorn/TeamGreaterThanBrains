# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:53:10 2016

@author: DTump
"""
#Calculates the average feature values per business for both train and testset.

import matplotlib.pyplot as plt
import PIL 
from PIL import Image
import os
import math
import pandas as pd
from pandas import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
import time 
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

#Load train data
print 'Loading Data....'
t = time.time()
##loads featurevectors
features1 = pd.read_csv('../../desktop/Features_data/caffe_features_train.csv',sep=',', header=None, iterator=True,chunksize=1000)
features =  concat(features1, ignore_index=True)
##loads links photo to business_id
train_photos = pd.read_csv('../../downloads/input/train_photo_to_biz_ids.csv',sep=',')
##loads business to labels 
biz_ids = pd.read_csv('../../downloads/input/train.csv',sep=',')
##loads order of featurevectors
photo_order = pd.read_csv('Features_data/photo_order_train.csv',sep=',',header=None)
X_TRAIN = pd.merge(biz_ids, train_photos, on='business_id')
print time.time()-t,
print 'seconds needed.'

#Calculates the averages per business
print 'Getting Business Averages....',
ti = time.time()
all_labels = [0] * 9
all_averages = [0] * 4096
t = 0
print 'Status: ',
for a in X_TRAIN.business_id.unique():                                  #for each business
    t = t+1
    data = X_TRAIN[(X_TRAIN.business_id == a)]
    labels = data['labels'].unique()
    if not isinstance(labels[0],str) and math.isnan(labels[0])==True:   #tocheckforbusinesses without labels
        continue
    labels = labels[0].split()
    labels = [int(x) for x in labels]
    labels = MultiLabelBinarizer().fit_transform([[0,1,2,3,4,5,6,7,8],labels])
    rev_labels = labels[1]
    all_labels = np.vstack((np.array(all_labels),np.array(rev_labels)))
    photos = data['photo_id']
    featavrg = [0] * 4096 
    for photo in photos:
            photo_ref = str(photo) + ''.join('m.jpg') 
            indx = photo_order.loc[photo_order[0] == photo_ref]
            if not(indx.empty):                                         #if no photos, delete from collection
                ind = indx.index[0]
                feats = features.loc[ind,:] 
                fts = feats.values.tolist()
                featavrg = np.vstack((np.array(featavrg),np.array(fts)))
    featavrg = featavrg[1:]
    averages = np.average(featavrg, axis=0)
    all_averages = np.vstack((np.array(all_averages),np.array(averages)))
    if t/100 == int(t/100):                  
        print t,
print ti - time.time(), 
print 'seconds needed.'

#Save averages and labels
np.savetxt("labels_train_y.csv", all_labels[1:], delimiter=",")
np.savetxt("feats_train_x.csv", all_averages[1:], delimiter=",")

#Load test data
t = time.time()
#loads featurevectors
features1 = pd.read_csv('../../desktop/Features_data/caffe_features_test.csv',sep=',', header=None, iterator=True,chunksize=1000)
features_test =  concat(features1, ignore_index=True)
#loads links photo to business_id
test_photos = pd.read_csv('../../downloads/input/test_photo_to_biz.csv',sep=',')
#loads order of featurevectors
photo_order_test = pd.read_csv('../../desktop/Features_data/photo_order_test.csv',sep=',',header=None)
print t-time.time()


#Calculate average feature values per business of testdata
ti = time.time()
businesses = [0]
all_averages = [0] * 4096
rejected_photos = [0]
t = 0
for a in test_photos.business_id.unique():                  #for each business
    t = t+1
    data = test_photos[(test_photos.business_id == a)]
    photos = data['photo_id']
    featavrg = [0] * 4096 
    for photo in photos:
            photo_ref = str(photo) + ''.join('m.jpg') 
            indx = photo_order_test.loc[photo_order_test[0] == photo_ref]
            if not(indx.empty):                             #if no photo, photo deleted from collection
                ind = indx.index[0]
                feats = features_test.loc[ind,:] 
                fts = feats.values.tolist()
                featavrg = np.vstack((np.array(featavrg),np.array(fts)))
            else:
                rejected_photos = np.vstack((np.array(rejected_photos),np.array(photo)))                
    featavrg = featavrg[1:]
    averages = np.average(featavrg, axis=0)
    all_averages = np.vstack((np.array(all_averages),np.array(averages)))
    businesses = np.vstack((np.array(businesses),np.array(a)))
    print t
print ti - time.time()

#Save averages, photo order and rejected photos
np.savetxt("biz_test.csv", businesses[1:], delimiter=",")
np.savetxt("rejected_photos_test.csv", rejected_photos[1:],delimiter = ",")
np.savetxt("feats_test_x.csv", all_averages[1:], delimiter=",")





