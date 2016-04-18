# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 20:00:34 2016

@author: Laurens
Follows clusteringVariationsKMAfterCluster, but with a different selection criteria for clustering of the eventual test set
"""

import numpy as np
import pandas as pd
import sys

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.cluster import MiniBatchKMeans

data = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in np.ones(4096)})

km = joblib.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Scripts/Clustering/512MBKM.pkl')
n_classes = 512

trainBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/trainSet.npy')
verifBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/verifSet.npy')


#=============== Prepare the clusteredTrainSet: ==================
photoToBiz = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv', sep=',')    
#photoToBiz = photoToBiz[~photoToBiz.business_id.isin(bizWithoutLabel)] #remove biz without a label
clusteredTrainSet = pd.DataFrame(index = trainBizIds, columns = range(n_classes))
clusteredVerifSet = pd.DataFrame(index = verifBizIds, columns = range(n_classes))
photoToIndex = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/photo_order_train.csv', sep=',', header=None).reset_index(level=0)

#--------- Cluster the train set
for busId in trainBizIds:
    photoIds = photoToBiz.loc[photoToBiz['business_id'] == busId].photo_id.to_frame()
    photoIds = photoIds.photo_id.map(lambda x: str(x)+ 'm.jpg')
    photoIds = photoIds.to_frame()
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index'] #Select only the indices of the images of this business
    bizPoints = data.iloc[featureIds.values]
    
    bizBools = np.zeros((bizPoints.shape[0],n_classes))
    clusters = km.predict(bizPoints)
    for i in range(bizPoints.shape[0]):
        bizBools[i][clusters[i]] = 1
    
    clusters = bizBools.sum(axis=0, dtype=np.float64)
    clusters /= clusters.max()
    clusteredTrainSet.loc[busId] = clusters
    
print('Clustered train set!')
clusteredTrainSet = clusteredTrainSet.sort_index()

#---------- Cluster the verification set
for busId in verifBizIds:
    photoIds = photoToBiz.loc[photoToBiz['business_id'] == busId].photo_id.to_frame()
    photoIds = photoIds.photo_id.map(lambda x: str(x)+ 'm.jpg')
    photoIds = photoIds.to_frame()
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index'] #Select only the indices of the images of this business
    bizPoints = data.iloc[featureIds.values]
    
    bizBools = np.zeros((bizPoints.shape[0],n_classes))
    clusters = km.predict(bizPoints)
    for i in range(bizPoints.shape[0]):
        bizBools[i][clusters[i]] = 1
    
    clusters = bizBools.sum(axis=0, dtype=np.float64)
    clusters /= clusters.max()
    clusteredVerifSet.loc[busId] = clusters
    
print('Clustered test set!')

#=================== Prepare the labelsTrainSet =========================
binLabels = pd.read_csv('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Scripts/binLabels.csv')
labelsTrainSet = binLabels[binLabels.business_id.isin(trainBizIds)]
labelsTrainSet = labelsTrainSet.sort_values('business_id')
del labelsTrainSet['business_id']
#----- fit SVM
#classifier = OneVsRestClassifier(SVC(kernel='poly')).fit(clusteredTrainSet.values, labelsTrainSet)
classifier = OneVsRestClassifier(LinearSVC()).fit(clusteredTrainSet.values, labelsTrainSet)
print('Fitted SVM')

#================= Calculate validation score ==========================
labelsVerifSet = binLabels[binLabels.business_id.isin(verifBizIds)]
labelsVerifSet = labelsVerifSet.sort_values('business_id')
del labelsVerifSet['business_id']

clusteredVerifSet = clusteredVerifSet.sort_index()
predictions = classifier.predict(clusteredVerifSet)

score = f1_score(labelsVerifSet.values.ravel(), predictions.ravel())

print('Score')
print score

f = open(str(n_classes) + 'Score2.txt', 'w')
f.write(str(score))
f.close()

