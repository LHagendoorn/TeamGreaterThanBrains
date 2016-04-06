# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 22:55:26 2016

@author: roosv_000
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
a=0
nrnan=0
ones=0
labelsload=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Submissions/submission2016-03-30testensemble44.csv', sep=',')
submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

alles=labelsload.values
labels=labelsload.ix[:,'labels':]
b=0
onelabel = [None] * 1000
for x in range(0, 10000):
    text=labels.at[x,'labels']
    
    if pd.isnull(text):
        nrnan=nrnan+1
        labels.at[x,'labels']='2 3 5 6 8'
    else:
        nrnumbers = sum(c.isdigit() for c in text)
        if nrnumbers==1:
            b=b+1
            onelabel[ones]=text
            ones=ones+1
            ind=x
            joinstrings=text+' 2 3 5 6 8'
            unique=' '.join(set(joinstrings))
            so=unique.split()
            so.sort()
            new = " ".join(so)
            labels.at[x,'labels']=new
            
        nums = [int(n) for n in text.split()]
        if 4 in nums and 6 in nums and 5 not in nums :
            a=a+1
        
        


submit['labels' ] = labels

#save in csv file
submit.to_csv('Ensembletest.csv',index=False)

            
