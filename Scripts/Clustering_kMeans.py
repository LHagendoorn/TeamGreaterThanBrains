# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:05:48 2016

@author: Laurens
"""

import numpy as np
from sklearn.cluster import KMeans
import scipy.io as sio

###############################################################################
# Load sample data
caffeatures = sio.loadmat('C:/Users/Laurens/Documents/TeamGreaterThanBrains/Features/caffe/caffe_features_werktdit.mat')['feats'].transpose()

###############################################################################
# Compute clustering with KMeans

km = KMeans(n_clusters=9)
km.fit(caffeatures)
labels = km.labels_
cluster_centers = km.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

###############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(caffeatures[my_members, 10], caffeatures[my_members, 11], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()