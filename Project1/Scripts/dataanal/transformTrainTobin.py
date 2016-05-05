# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 12:51:45 2016

@author: Laurens
Takes the actual labels for each business and transforms it into a binary representation.
"""

import pandas as pd
import numpy as np

def comp(lbls):
    result = np.zeros(9)
    for i in range(9):
        if i in lbls:
            result[i] = 1
    return result

train = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train.csv', sep=',')
trainBin = pd.DataFrame(index=train.business_id, columns = range(9))

for busId in train.business_id:
    lbls = train[train.business_id == busId]['labels'].values[0]
    if lbls != lbls:
        lbls = ''
    trainBin.loc[busId] = comp(map(int, str.split(lbls)))