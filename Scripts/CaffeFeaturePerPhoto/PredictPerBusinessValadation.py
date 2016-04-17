# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 22:45:42 2016

@author: roosv_000
Labels per photo are predited with an SVM, this script uses these prediced labels per photo to create a vector (normalized histogram) 
respresenting the label predictions per business. The business feature vectors are in the output dataframe.
It does this for all businesses in the Valadationset.
"""
import numpy as np
import pandas as pd

threshold=0.5

PhotoBusid=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/train_photo_to_biz_ids.csv', sep=';')
#LabelsForPhotosVal=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/pred_testSetPerImage.csv', sep=',')

LabelsForPhotosVal=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/pred_ValSetPerImage_trainedontrainvantrain.csv', sep=',')

LabelsForPhotosVal.drop(LabelsForPhotosVal.columns[[0]], axis=1, inplace=True)
LabelsForPhotosVal['photo_id'] = LabelsForPhotosVal['photo_id'].map(lambda x: str(x)[:-5]) #remove m.jpg
LabelsForPhotosVal['photo_id'] = LabelsForPhotosVal['photo_id'].astype(int) #set type of photoids to int

veribus=np.load('../../verifSet.npy')
UniqueBus=veribus
NrUniqueBus=len(UniqueBus)
NrPhotos=len(PhotoBusid['photo_id'])
data=np.zeros((NrUniqueBus,9))

for x in range(0,NrUniqueBus):
    busid=UniqueBus[x]
    ind,=np.where(PhotoBusid['business_id']==busid)
   
    photoids=PhotoBusid['photo_id'][ind].astype(int) #get photo ids from current busness
    photoids = pd.np.array(photoids)
    mask = LabelsForPhotosVal[['photo_id']].isin(photoids).all(axis=1)
    allphotolabels=LabelsForPhotosVal.ix[mask]
    labelsarray= allphotolabels.drop('photo_id', 1).values
    sumlabels=np.sum(labelsarray,axis=0)
    maxnrlabels=np.amax(sumlabels)
    floatsumlabels = np.array(sumlabels, dtype=float)
    if maxnrlabels==0:
        normhist=floatsumlabels
    else:
        normhist=floatsumlabels/maxnrlabels
        
    data[x]=normhist
    print(x)

#Make the stucture for output dataframe
indexx=np.arange(NrUniqueBus)
dtype = [('business_id','int64'), ('0','float64'),('1','float64'),('2','float64'),('3','float64'),('4','float64'),('5','float64'),('6','float64'),('7','float64'),('8','float64')]
values = np.zeros(NrUniqueBus, dtype=dtype)
output = pd.DataFrame(values, index=indexx)  

#put UniqueBusinesses and normalized label histogram in a dataframe
output['business_id']=UniqueBus
output[['0','1','2','3', '4','5','6','7','8']]=data


  
    
 
