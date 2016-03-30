# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:57:34 2016

@author: Laurens
"""

import numpy as np
import pandas as pd

from sklearn.mixture import GMM
from sklearn.externals import joblib

photoToBiz = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv', sep=',')    
busIds = pd.unique(photoToBiz.business_id.ravel())

df = pd.DataFrame(index = busIds, columns = range(100))

classifier = joblib.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/clusters/100clustersGMMEM/100clGMMEM.pkl')

testRead = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, nrows = 1)
caffeatures = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in list(testRead)})

photoToIndex = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/photo_order_train.csv', sep=',', header=None).reset_index(level=0)
count = busIds.size;

for busId in busIds:
    photoIds = photoToBiz.loc[photoToBiz['business_id'] == busId].photo_id.to_frame()
    photoIds = photoIds.photo_id.map(lambda x: str(x)+ 'm.jpg')
    photoIds = photoIds.to_frame()
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index']
    probs = classifier.predict_proba(caffeatures.iloc[featureIds.values])
    probs = probs.sum(axis=0, dtype=np.float64)
    probs /= probs.max()
    df.loc[busId] = probs
    count-=1
    print(count)
    if (busIds.size-count)%500==0:
            df.to_csv('100GMMEMHistogramProbsTrain.csv')
    
df.to_csv('100GMMEMHistogramProbsTrain.csv')
