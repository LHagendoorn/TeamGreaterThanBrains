# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:15:51 2016

@author: roosv_000
"""
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time 

submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

PhotoBusid=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/train_photo_to_biz_ids.csv', sep=';')

testhist=np.load('../Labels per photo/data_array_val_hist_9-4_OnTrainset.npy')
trainhist=np.load('../Labels per photo/data_array_train_hist_9-4_OnTrainset.npy')

trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')

trainbus=np.load('../../trainSet.npy')
trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')

masklabels = trainlabels[['business_id']].isin(trainbus).all(axis=1)
traintrain=trainlabels.ix[masklabels]
trainlab=traintrain.sort_values(by='business_id')
df = pd.DataFrame(trainhist)
df=df.ix[masklabels]
df=pd.DataFrame.as_matrix(df)

#trainhist.drop(trainhist.index[masklabels])

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
trainlabelsbool = trainlab['labels'].apply(to_bool)
trainlabelsbool=pd.DataFrame.as_matrix(trainlabelsbool)
print 'done intitializing data'

#TRAIN SVM
print 'Training SVM....'   
ti = time.time()

S = OneVsRestClassifier(SVC(kernel='poly',probability=True)).fit(df, trainlabelsbool)
#S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainhist, trainlabelsbool)

score = S.score(df,trainlabelsbool)
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

print t-time.time()

bla=S.predict_proba(testhist)
#bla=S.predict(testhist)



