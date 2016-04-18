# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:15:59 2016

@author: roosv_000

Ensemble of 5 models is tested on a valadationset to determine the desired weight for each model. 
Label probalilities for the valadation businesses predicted by five different models are imported.
The F1 scores for the ensemble of models is calculated 

MODELS:
- Statistical model, the probability of a label occurring for a business.
- SVM mean model, takes the mean caffe-features per business, and uses an SVM to classify.
- Color Feature model, takes mean Color feature per business, and uses Random Forest to classify.
- Per Photo SVM model, predict labels per photo with SVM of caffe features than another SVM to predict per business.
- Cluster model 

set:
-model directories
-filename
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

filename= 'Ensemble_Tuned_On_Val_Set_With_SVM.csv'
#import models for valitdation set, every model consist of a matrix label probabilities for all businesses in the val set. 
model1 = pd.read_csv('probSTAT.csv', sep=';', header=None).values
model2load= pd.read_csv('../SVM/SVM_verifset_mean1punt0.csv',sep=',',index_col=2)
model3load= pd.read_csv('../randomForest/probColorValidationSetWithIds.csv', sep=',', header=None,index_col=0)
model4 = np.load('../Labels per photo/Prob_Valset_FromHistSVM.npy')
model5= np.load('MBKMpredictions128.npy')

#load models, every model consist of a matrix label probabilities for all businesses in the test set.
test_prob_model1=pd.read_csv('probSTAT.csv', sep=';', header=None).values
test_prob_model2=pd.read_csv('probSVM.csv',sep=',', header=None).values
test_prob_model3= pd.read_csv('probColor.csv',sep=',', header=None).values
test_prob_model4=np.load('../Labels per photo/data_array_test_9-4.onallprob.npy')
test_prob_model5= np.load('MBKMpobabilitiesTest128.npy')

#adjust model size of model 1
valbus=np.load('../../verifSet.npy')
model1 = np.delete(model1, (range(9800)), axis=0)
model2 = model2load.drop('index.1', 1)
model2 = model2.drop('index', 1)
model2 = model2.drop('Unnamed: 0', 1)
model2 = model2.reindex(valbus).values
model3=model3load.reindex(valbus).values

#import the example submission file for its stucture
submit = pd.read_csv('SubmissionFormat.csv',sep=',')

#Concatenate the probabilies for valadation set from de different models.
ConcatModels=np.concatenate((model1,model2,model3,model4,model5),axis=1)

#Concatenate the probabilies for valadation set from de different models.
ConcatTestProb=np.concatenate((test_prob_model1,test_prob_model2,test_prob_model3,test_prob_model4, test_prob_model5),axis=1)


#get true labels for valset
ytruelab=[]
trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')
valbus=np.load('../../verifSet.npy')
for x in range(len(valbus)):
    currbus= valbus[x]
    roww = trainlabels.loc[trainlabels['business_id']==(currbus)]
    lab=roww['labels'].values[0]
    ytruelab.append(lab)
    
# convert numeric labels for the true valadation labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))

ytruelabSeries=pd.Series(ytruelab)
ytrue = ytruelabSeries.apply(to_bool)    

#SVM on the probabilities for all models (ConcatModels) and true label values for the valadationset(ytrue)
#S = OneVsRestClassifier(SVC(kernel='poly')).fit(model2, ytrue)
S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(ConcatModels, ytrue)

score = S.score(ConcatModels,ytrue)
PredictTestLabels=S.predict(ConcatTestProb)

#convert array ensembleprob [0 1 0 0 0 1] to list of strings ['2 6']
predList = []
for row in PredictTestLabels:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        labelstr = sep.join(indices)
        predList.append(labelstr)

# Put labels(predList) in de submissionfile colomn named 'labels'
submit['labels' ] = predList

#save in csv file
submit.to_csv(filename,index=False)


