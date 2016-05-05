# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:45:00 2016

@author: roosv_000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv',sep=';')
biz_id_train = pd.read_csv('train_photo_to_biz_ids.csv',sep=';')
submit = pd.read_csv('sample_submission.csv')

# plot distribution of photo counts
photo_count = biz_id_train.groupby('business_id')
photo_count2 = photo_count.count()
photo_count3 = photo_count2.sort_values('photo_id')
numberofphotos=photo_count3.hist(bins=3000)
mat=biz_id_train.as_matrix
uniek=np.unique(mat)

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()
plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# plot correlation of outputs
correlation=Y.corr()
plt.matshow(Y.corr(),cmap=plt.cm.RdBu)
plt.colorbar()

# 3 (outdoor_seating) is rather uncorrelated with the rest
# 0 (good_for_lunch) negatively correlated with the other descriptors, except good for kids
# 1,2,4-7 are a correlated cluster

# simulate randomly based on mean proportions
np.random.seed(290615)
submit['labels'] = submit.apply(lambda x: ' '.join( \
[str(i) for i in np.where(np.random.binomial(1,py,size=(9)))[0]]),axis=1)
submit.to_csv('sub1_naive.csv',index=False)