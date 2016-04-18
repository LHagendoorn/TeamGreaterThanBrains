# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:05:48 2016

@author: Laurens

Mean shift attempt, turns out this was too slow/memory intesive to be practical
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd


###############################################################################
"""Load sample data"""
#caffeatures = sio.loadmat('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Features/caffe/caffe_features_werktdit.mat')['feats']
#caffeatures = caffeatures.transpose()
testRead = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/caffe_features_train.csv', header=None, nrows = 1)
caffeatures = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in list(testRead)})

###############################################################################
""" Compute clustering with MeanShift"""

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(testRead, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(caffeatures)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

###############################################################################
""" Plot result """
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(caffeatures[my_members, 20], caffeatures[my_members, 30], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()