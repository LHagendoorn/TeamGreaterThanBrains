# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:15:59 2016

@author: roosv_000

Ensemble of 4 models is tested on a valadationset to determine the desired weight for each model. 
Label probalilities for the valadation businesses predicted by four different models are imported.
The F1 scores for the ensemble of models is calculated 
set:
-model directories
-weights
"""

import numpy as np
import pandas as pd
import sklearn.metrics

model1 = pd.read_csv('probSTAT.csv', sep=';', header=None).values
model2= np.load('../Labels per photo/Prob_Valset_FromHistSVM.npy')
model3= np.load('../Labels per photo/Prob_Valset_FromHistSVM.npy')
model4 = np.load('../Labels per photo/Prob_Valset_FromHistSVM.npy')

#weights for the models, if you only want to ensemble 2 methods set the third and forth value on 0
weights = [0.25, 0.75, 0, 0]

#classification threshold
threshold = 0.5

#adjust model size of model 1
model1 = np.delete(model1, (range(9800)), axis=0)

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

#Create one matrix with the probability for each label for every business, by combining the probabilities in the models with certain weights.
ensembleprob = model1 * weights[0] + model2 * weights[1] + model3 * weights[2] + model4 * weights[3]

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
        
#calculate F1 score
Average_F1=sklearn.metrics.f1_score(ytrue, ensembleprob, average='weighted')
print('Average F1 score=', Average_F1)

