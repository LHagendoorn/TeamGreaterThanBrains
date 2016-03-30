# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:53:10 2016

@author: DanielleT
@author: Diede
"""

import matplotlib.pyplot as plt
import PIL 
from PIL import Image
import os
import math
import LoadData
import pandas as pd
from pandas import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC






#LOAD DATA
print 'Loading Data....'
generalData = LoadData.load()
#generalData = dictionary with fields X_TRAIN, Y_TRAIN and X_TEST
trainFeatureData = LoadData.load_caffe_features(trainSet=True)
#trainFeatureData = dictionary with fields FEATURES and ORDER
print 'Loading data is done.'


#Summarize data per business



'''
Gives the aggregated features per business (the Median, Q1, Q3, Min and Max)

Xtrain, features, photo_order all are dataFrames.

-Uses the photo_order data to know which feature matches which photoID,
-Uses the Xtrain data to know which business has which photoIDs

Returns: nrOfBusinesses x nrOfFeatures x 5 list! (not dataframe)
'''
def get_aggregated_data(Xtrain, features, photo_order):

    #initialize
    counter = 0
    businessIndex = 0
    allBusinessIDs = Xtrain.business_id.unique()
    nrOfBusinesses = len(allBusinessIDs)
    nrOfFeatures = features.shape[1]

    #pre-allocate space for aggregated data
    aggregatedData = [0] * nrOfBusinesses * nrOfFeatures * 5

    #for each business
    for businessID in allBusinessIDs:
        #extract photoIDs of this business
        ff = Xtrain[(Xtrain.business_id == businessID)]
        photosIDs = ff['photo_id']
        nrOfPhotos = len(photosIDs)

        #pre-allocate space for feature vectors of this business
        businessData = [0] * nrOfPhotos * nrOfFeatures

        #count how many photosIDS did not have a feature vector
        nrOfMissingPhotos = 0
        #keep track of index to place the feature vector
        photoIndex = 0

        #walk through the photoIDs
        for photoID in photosIDs:

            #find index of this photo in the features
            photo_ref = str(photoID) + ''.join('m.jpg')
            indx = photo_order.loc[photo_order[0] == photo_ref]

            #check whether photo has a feature vector. If so:
            if not(indx.empty):
                #save the feature vector of this photo
                ind = indx.index[0]
                feats = features.loc[ind,:]
                businessData[photoIndex][:] = feats.values.tolist()
                photoIndex += 1

            else: ##count the missing photo
                nrOfMissingPhotos += 1

        #remove rows of missing photos
        businessData = businessData[:-nrOfMissingPhotos][:]

        #compute and save aggregated data
        aggregatedData[businessIndex][:][:] = aggregate(businessData)
        businessIndex += 1

        #print progress
        counter = counter + 1
        if counter/100 == int(counter/100):
            print(counter)


'''
Gives the Median, Q1, Q3, Min and Max of the given featureVectors

featureVectors = list of size nrOfPhotos x nrOfFeatures

Returns: list of size nrOfFeatures x 5
'''
def aggregate(featureVectors):
    df = pd.DataFrame(featureVectors)

    laag = df.min()
    q1 = df.quantile(0.25)
    q2 = df.quantile(0.5)
    q3 = df.quantile(0.75)
    hoog = df.max()

    print('debug: deze waardes moeten 4096 zijn')
    print(len(laag))
    print(len(q1))
    print(len(q2))
    print(len(q3))
    print(len(hoog))

    dinges = [laag, q1, q2, q3, hoog]
    result = pd.concat(dinges, axis=1)

    return result.values



np.savetxt("labels_train_y.csv", all_labels[1:], delimiter=",")
np.savetxt("feats_train_x.csv", all_averages[1:], delimiter=",")

#toepassen SVM 
print 'Training SVM....'   
y = all_labels[1:]
x = all_averages[1:]
S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y)
score = S.score(x,y)
print score
#
#
#
#
##-------------------------------PRODUCE TEST DATA
#loads featurevectors
features1 = pd.read_csv('../../desktop/Features_data/caffe_features_test.csv',sep=',', header=None, iterator=True,chunksize=1000)
features_test =  concat(features1, ignore_index=True)
#loads links photo to business_id
test_photos = pd.read_csv('../../downloads/input/test_photo_to_biz.csv',sep=',')
#loads order of featurevectors
photo_order_test = pd.read_csv('../../desktop/Features_data/photo_order_test.csv',sep=',',header=None)


businesses = [0]
all_averages = [0] * 4096
rejected_photos = [0]
t = 0
for a in test_photos.business_id.unique():          #for each business
    t = t+1
    data = test_photos[(test_photos.business_id == a)]
    photos = data['photo_id']
    featavrg = [0] * 4096 
    for photo in photos:
            photo_ref = str(photo) + ''.join('m.jpg') 
            indx = photo_order_test.loc[photo_order_test[0] == photo_ref]
            if not(indx.empty):                     #photo deleted from collection
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

#save averages
np.savetxt("biz_test.csv", businesses[1:], delimiter=",")
np.savetxt("rejected_photos_test.csv", rejected_photos[1:],delimiter = ",")
np.savetxt("feats_test_x.csv", all_averages[1:], delimiter=",")

##Perform
x = all_averages[1:]

predictions = S.predict(x)
print predictions[:5]
np.savetxt("predictions_test.csv", predictions, delimiter=",")

#print outputfile
bizz = test_photos.business_id.unique()
a=0
arr = [0]*2
while True:
    strlabels = ""
    for i in range(0,8):
        pred = predictions[a]
        lab = pred[i]
        if lab==1:
            strlabels = strlabels + " " + str(i)
    currarr = [bizz[a],strlabels.lstrip()]    
    arr = np.vstack((np.array(arr),np.array(currarr)))
    if bizz[a]==bizz[-1]:
        break
    a = a+1

arr[0]=['business_id','labels']

np.savetxt("subm.csv", arr, delimiter=",")

this = pd.DataFrame(data=arr[1:,:],columns=arr[0,:])  # 1st row as the column names
this.to_csv('feats_sub3.csv',index=False)





