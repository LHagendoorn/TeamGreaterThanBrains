# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:05:48 2016

@author: Laurens
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
#import scipy.io as sio
import pandas as pd

###############################################################################
# Load sample data
#caffeatures = sio.loadmat('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Features/caffe/caffe_features_werktdit.mat')['feats'].transpose()
testRead = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/caffe_features_train.csv', header=None, nrows = 1)
caffeatures = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in list(testRead)})


###############################################################################
# Compute clustering with KMeans

print('HOOOOOI')
#km = KMeans(n_clusters=100, n_jobs = -2)
km = MiniBatchKMeans(n_clusters=100)
km.fit(caffeatures)
labels = km.labels_
cluster_centers = km.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)



###############################################################################