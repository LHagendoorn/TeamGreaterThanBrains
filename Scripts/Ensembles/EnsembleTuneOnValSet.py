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
-weights
"""

import numpy as np
import pandas as pd
import sklearn.metrics


#load models, every model consist of a matrix label probabilities for all businesses in the valadation setset.
model1 = pd.read_csv('probSTAT.csv', sep=';', header=None).values
model2load= pd.read_csv('../SVM/SVM_verifset_mean1punt0.csv',sep=',',index_col=2)
model3load= pd.read_csv('../randomForest/probColorValidationSetWithIds.csv', sep=',', header=None,index_col=0)
model4 = np.load('../Labels per photo/Prob_Valset_FromHistSVM.npy')
model5= np.load('MBKMpredictions128.npy')

#Load businesses in valadation set
valbus=np.load('../../verifSet.npy')

#weights for the models, if you only want to ensemble 2 methods set other weights to zeros (sum of weight have to be 1)
weights = [0, 0.6, 0, 0.4, 0]

#classification threshold
threshold = 0.5

#adjust models format
model1 = np.delete(model1, (range(9800)), axis=0)
model2 = model2load.drop('index.1', 1)
model2 = model2.drop('index', 1)
model2 = model2.drop('Unnamed: 0', 1)
model2 = model2.reindex(valbus).values
model3=model3load.reindex(valbus).values

#get true labels for valset
ytruelab=[]
trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')
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

#Create one matrix with the probability for each label for every business, by combining the probabilities in the models with certain weights.
ensembleprob = model1 * weights[0] + model2 * weights[1] + model3 * weights[2] + model4 * weights[3] + model5 * weights[4]

#Classify by converting probabilities into binaries with help of the threshold.
ensembleprob[ensembleprob > threshold] = 1
ensembleprob[ensembleprob <= threshold] = 0

#convert array ensembleprob [0 1 0 0 0 1] to list of strings ['2 6']
predList = []
for row in ensembleprob:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        labelstr = sep.join(indices)
        predList.append(labelstr)

#Calculate F1 score for the seperate labels       
for WhichLabel in range(9):
    Average_F1=sklearn.metrics.f1_score(ytrue[WhichLabel].values, ensembleprob[:,WhichLabel])
    print(''.join(['F1 score for label ',str(WhichLabel),' = ', str(Average_F1)]))

#Calculate the total weighted F1 score for the valadation set
Total_Average_F1=sklearn.metrics.f1_score(ytrue, ensembleprob, average='weighted')
print(weights)

   # print(''.join(['F1 score for label ',str(WhichLabel),' = ', str(Average_F1)]))
print('')
print(''.join(['Total_Average F1 score = ', str(Total_Average_F1)]))

    
    
