# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 18:18:32 2016

@author: Laurens
"""

import numpy as np
import pandas as pd
import sys

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.cluster import MiniBatchKMeans

bizWithoutLabel = [1627, 2661, 2941, 430] #shockingly enough, hardcoding is the most effiient way I can come up with at the moment to not have to load an entirely new csv file

#REMOVE ROW LIMIT WHEN NOT TESTING
#testRead = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, nrows = 1)
trainData = pd.read_csv('$HOME/features/Features_data/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in np.ones(4096)})
data = pd.concat([trainData, pd.read_csv('$HOME/features/Features_data/caffe_features_test.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in np.ones(4096)})])
print('data loaded!')

#dependend on the instantiation
classOptions = [2048, 1024, 512, 256, 128, 64]
#covOptions = ['diag', 'spherical', 'tied', 'full']
arg = int(sys.argv[1])
n_classes = classOptions[arg]
#cov_type = covOptions[arg%4]

print(str(n_classes))

km = MiniBatchKMeans(n_clusters = n_classes,batch_size = 300, verbose = True)
pointsToClusterDist = km.fit_transform(data)
meanDist = np.mean(pointsToClusterDist,axis=0)
cluster_centers = km.cluster_centers_

print 'fitted & transformed!'

joblib.dump(km, str(n_classes) + 'MBKM.pkl')
print 'saved model!'
#clusterer = GMM(n_components=n_classes, covariance_type=cov_type,n_iter=0, init_params = 'wc')
#clusterer.means_ = cluster_centers
#clusterer.fit(data)
#bicScore = clusterer.bic(data)

#clusterer.set_params(n_iter=1,init_params='')
#prevBicScore = bicScore+2
#while abs(prevBicScore - bicScore) >1:
#    clusterer.fit(data)
#    prevBicScore = bicScore
#    bicScore = clusterer.bic(data)
#    print('bicScore:')    
#    print(bicScore)
#    print('diff')
#    print(prevBicScore-bicScore)

#save the clusters in case it's the best:
#joblib.dump(clusterer, str(n_classes) + cov_type + 'GMMEM.pkl')
#print('Saved the fitted clusterer')

#get the businessId's for the train and verification set (not including the empty labels)
#trainBizIds, verifBizIds = getSplit()
trainBizIds = np.load('$HOME/trainSet.npy')
verifBizIds = np.load('$HOME/verifSet.npy')


#=============== Prepare the clusteredTrainSet: ==================
photoToBiz = pd.read_csv('$HOME/train_photo_to_biz_ids.csv', sep=',')    
photoToBiz = photoToBiz[~photoToBiz.business_id.isin(bizWithoutLabel)] #remove biz without a label
clusteredTrainSet = pd.DataFrame(index = trainBizIds, columns = range(n_classes))
clusteredVerifSet = pd.DataFrame(index = verifBizIds, columns = range(n_classes))
photoToIndex = pd.read_csv('$HOME/features/Features_data/photo_order_train.csv', sep=',', header=None).reset_index(level=0)

#--------- Cluster the train set
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

#---------- Cluster the verification set
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
binLabels = pd.read_csv('$HOME/labels_train_y.csv', header=None)
indexToBizId = pd.DataFrame({'bizId': photoToBiz.business_id.unique()}).reset_index(level=0)
trainIndexToBizId = indexToBizId[indexToBizId.bizId.isin(trainBizIds)]
labelsTrainSet = binLabels.iloc[trainIndexToBizId['index']]

#----- fit SVM
classifier = OneVsRestClassifier(SVC(kernel='poly')).fit(clusteredTrainSet.values, labelsTrainSet)

print('Fitted SVM')

#================= Calculate validation score ==========================
verifIndexToBizId = indexToBizId[indexToBizId.bizId.isin(verifBizIds)]
labelsVerifSet = binLabels.iloc[verifIndexToBizId['index']]

predictions = classifier.predict(clusteredVerifSet)

score = f1_score(labelsVerifSet.values.ravel(), predictions.ravel())

print('Score')
print score

f = open(str(n_classes) + 'Score.txt', 'w')
f.write(str(score))
f.close()

