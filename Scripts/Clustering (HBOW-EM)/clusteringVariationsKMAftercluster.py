# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 18:18:32 2016

@author: Laurens

Loads in a trained clusterer and generates a score based on selecting each of
the clusters that are closer than the mean distance
"""

import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.cluster import MiniBatchKMeans

#REMOVE ROW LIMIT WHEN NOT TESTING
data = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in np.ones(4096)})
print 'data loaded!!'

km = joblib.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Scripts/Clustering/1024MBKM.pkl')
n_classes = 1024
print 'clusterer loaded!'
pointsToClusterDist = km.transform(data)
meanDist = np.mean(pointsToClusterDist,axis=0)

#get the businessId's for the train and verification set (not including the empty labels)
trainBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/trainSet.npy')
verifBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/verifSet.npy')


#=============== Prepare the clusteredTrainSet: ==================
photoToBiz = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv', sep=',')    
clusteredTrainSet = pd.DataFrame(index = trainBizIds, columns = range(n_classes))
clusteredVerifSet = pd.DataFrame(index = verifBizIds, columns = range(n_classes))
photoToIndex = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/photo_order_train.csv', sep=',', header=None).reset_index(level=0)

#--------- Cluster the train set
print 'clustering train set:'
for busId in trainBizIds:
    photoIds = photoToBiz.loc[photoToBiz['business_id'] == busId].photo_id.to_frame()
    photoIds = photoIds.photo_id.map(lambda x: str(x)+ 'm.jpg')
    photoIds = photoIds.to_frame()
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index'] #Select only the indices of the images of this business
    bizPoints = data.iloc[featureIds.values]
    bizPointsDists = km.transform(bizPoints)
    #MAYBE TEST VARIOUS SELECTION CRITERIA
    bizBools = bizPointsDists[:] < meanDist
    clusters = bizBools.sum(axis=0, dtype=np.float64)
    clusters /= clusters.max()
    clusteredTrainSet.loc[busId] = clusters
    
print('Clustered train set!')
clusteredTrainSet = clusteredTrainSet.sort_index()


#---------- Cluster the verification set
print 'clustering test set:'
for busId in verifBizIds:
    photoIds = photoToBiz.loc[photoToBiz['business_id'] == busId].photo_id.to_frame()
    photoIds = photoIds.photo_id.map(lambda x: str(x)+ 'm.jpg')
    photoIds = photoIds.to_frame()
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index'] #Select only the indices of the images of this business
    bizPoints = data.iloc[featureIds.values]
    bizPointsDists = km.transform(bizPoints)
    #MAYBE TEST VARIOUS SELECTION CRITERIA
    bizBools = bizPointsDists[:] < meanDist
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

score = f1_score(labelsVerifSet, predictions, average='weighted')

print('Score')
print score

f = open(str(n_classes) + 'ScoreCorrectOrderWeighted.txt', 'w')
f.write(str(score))
f.close()

