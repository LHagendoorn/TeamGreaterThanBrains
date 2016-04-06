# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 22:45:42 2016

@author: roosv_000
"""
import numpy as np
import pandas as pd
#import collections


PhotoBusid=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/train_photo_to_biz_ids.csv', sep=';')
LabelsForPhotos=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/photolabeltest2.csv', sep=';')

UniqueBus=np.unique(PhotoBusid['business_id'])
NrUniqueBus=len(UniqueBus)
NrPhotos=len(PhotoBusid['photo_id'])
data=np.zeros((NrUniqueBus,9))

for x in range(0,1000):
#for x in range(0,NrUniqueBus):
    busid=UniqueBus[x]
    index,=np.where(PhotoBusid['business_id']==busid)
   
    photoids=PhotoBusid['photo_id'][index] #get photo ids from current busness
 
    mask = LabelsForPhotos[['photo_id']].isin(photoids).all(axis=1)
    allphotolabels=LabelsForPhotos.ix[mask]
    labelsarray= allphotolabels.drop('photo_id', 1).values
    sumlabels=np.sum(labelsarray,axis=0)
    maxnrlabels=np.amax(sumlabels)
    floatsumlabels = np.array(sumlabels, dtype=float)
    if maxnrlabels==0:
        normhist=floatsumlabels
    else:
        normhist=floatsumlabels/maxnrlabels
        
    data[x]=normhist
    
#Make the stucture for output dataframe
indexx=np.arange(NrUniqueBus)
dtype = [('business_id','int64'), ('0','float64'),('1','float64'),('2','float64'),('3','float64'),('4','float64'),('5','float64'),('6','float64'),('7','float64'),('8','float64')]
values = np.zeros(NrUniqueBus, dtype=dtype)
output = pd.DataFrame(values, index=indexx)  

#put UniqueBusinesses and normalized label histogram in a dataframe
output['business_id']=UniqueBus
output[['0','1','2','3', '4','5','6','7','8']]=data

#select only the Train businesses without the Valadation businesses.

  
    
 
