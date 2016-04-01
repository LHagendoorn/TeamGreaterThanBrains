# -*- coding: utf-8 -*-
"""
Gives a stratified split of the train data
Created on Fri Apr 01 13:51:40 2016

@author: Laurens
"""
import pandas as pd
import numpy as np
from  sklearn.cross_validation import StratifiedShuffleSplit

def getSplit() :
    bizLabels = pd.read_csv('c:/Users/Laurens/documents/uni/MLP/data/train.csv', sep=',')
    bizLabels.columns = ['busId','lbls'] #labels is a reserved keyword!!
    #photoToBiz = pd.read_csv('c:/Users/Laurens/documents/uni/MLP/data/train_photo_to_biz_ids.csv', sep=',')
    
    nBizPerLabel = bizLabels.groupby('lbls').count()
    nBizPerLabel.columns = ['bizCounts']
    
    #nPhotoPerBiz = photoToBiz.groupby('business_id').count()
    #nPhotoPerBiz.columns = ['photoCounts']
    
    includedLabels = nBizPerLabel.loc[nBizPerLabel['bizCounts']>1]
    excludedLabels = nBizPerLabel.loc[nBizPerLabel['bizCounts']==1]
    bizLabelsIncluded = bizLabels[bizLabels.lbls.isin(includedLabels.index)]
    bizLabelsExcluded = bizLabels[bizLabels.lbls.isin(excludedLabels.index)]
    #We are going for a tenth of the train set for validation (200 businesses), 
    #52 businesses are excluded from the stratifiedSplit, yet we would like to include at least a few,
    #52 is 2.6% of 2000, so it should be 2.6% of the sample which is 5.2, so let's say 5, 195 remain 
    #but somehow a sample size of 195 does not return 195 samples, 201 does??????
    strSplit = StratifiedShuffleSplit(bizLabelsIncluded['lbls'].values, n_iter=1,test_size=201, train_size=None)
    train=[]
    verif=[]
    for trainIndex, verifIndex in strSplit:  
        s = bizLabelsExcluded.busId.sample(5)
        train = bizLabelsIncluded.busId.iloc[trainIndex].values
        verif = bizLabelsIncluded.busId.iloc[verifIndex].values
        verif = np.append(verif, s)
        train = np.append(train, bizLabelsExcluded[~bizLabelsExcluded.busId.isin(s)].busId.values)
        #uncomment to include the non-labeled entries:
        #train = np.append(train, bizLabels.loc[bizLabels['lbls'].isnull()].busId.values) #these have no label, and a very slim chance to end up in the validation set anyway, so let's just stick them in the train set
    return train, verif