# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:03:55 2016

@author: Laurens
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
print 'data loaded!!'

#Load the best clusterer
km = joblib.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Scripts/Clustering/128MBKM.pkl')
n_classes = 128
print 'clusterer loaded!'

pointsToClusterDist = km.transform(data)
meanDist = np.mean(pointsToClusterDist,axis=0)

trainBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/trainSet.npy')
verifBizIds = np.load('C:/Users/Laurens/Documents/TeamGreaterThanBrains/verifSet.npy')

#================================== calculate varification set probabilities for ensemble tweeking

#--- train on train set of train data
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
    featureIds = pd.merge(photoToIndex, photoIds, left_on=0, right_on='photo_id')['index']
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
classifier = OneVsRestClassifier(SVC(kernel='poly', probability = True)).fit(clusteredTrainSet.values, labelsTrainSet)
#classifier = OneVsRestClassifier(LinearSVC()).fit(clusteredTrainSet.values, labelsTrainSet)
clusteredVerifSet = clusteredVerifSet.sort_index()
predictions = classifier.predict_proba(clusteredVerifSet)



totalTrainData = clusteredTrainSet.append(clusteredVerifSet)

labelsVerifSet = binLabels[binLabels.business_id.isin(verifBizIds)]
labelsVerifSet = labelsVerifSet.sort_values('business_id')
del labelsVerifSet['business_id']

totalLabels = np.append(labelsTrainSet,labelsVerifSet, axis=0)
#classifier = OneVsRestClassifier(SVC(kernel='poly', probability = True)).fit(totalTrainData.values, totalLabels)
classifier = OneVsRestClassifier(LinearSVC()).fit(clusteredTrainSet.values, labelsTrainSet)


data = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_test.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in np.ones(4096)})

photoToBiz = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/test_photo_to_biz.csv', sep=',')    
testBizIds = photoToBiz.business_id.unique()

photoToIndex = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/photo_order_test.csv', sep=',', header=None).reset_index(level=0)
clusteredTestSet = pd.DataFrame(index = testBizIds, columns = range(n_classes))

#---------- Cluster the test data
print 'clustering test data:'
for busId in testBizIds:
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
    clusteredTestSet.loc[busId] = clusters
    
print('Clustered test set!')

probabilitiesTest = classifier.predict_proba(clusteredTestSet)
predictionsTest = classifier.predict(clusteredTestSet)



#import the example submission file for its stucture
submit = pd.read_csv('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

filename='128linearSubmission.csv'


#input for this has to be an array in the order of the testbusinesses in the submissionfile
#convert array ensembleprob [0 1 0 0 0 1] to list of strings ['2 6']
predList = []
for row in predictionsTest:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        labelstr = sep.join(indices)
        predList.append(labelstr)

# Put labels(predList) in de submissionfile colomn named 'labels'
submit['labels' ] = predList

#save in csv file
submit.to_csv(filename,index=False)