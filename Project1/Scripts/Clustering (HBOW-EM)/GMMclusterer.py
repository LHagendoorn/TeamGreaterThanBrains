"""
==================
GMM classification
==================

Demonstration of Gaussian mixture models for classification.

See :ref:`gmm` for more information on the estimator.

Plots predicted labels on both training and held out test data using a
variety of GMM classifiers on the iris dataset.

Compares GMMs with spherical, diagonal, full, and tied covariance
matrices in increasing order of performance.  Although one would
expect full covariance to perform best in general, it is prone to
overfitting on small datasets and does not generalize well to held out
test data.

On the plots, train data is shown as dots, while test data is shown as
crosses. The iris dataset is four-dimensional. Only the first two
dimensions are shown here, and thus some points are separated in other
dimensions.
"""
import numpy as np
import pandas as pd

from sklearn.mixture import GMM
from sklearn.externals import joblib

testRead = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, nrows = 1)
caffeatures = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/features/caffe_features_train.csv', header=None, sep=',', engine='c', dtype={c: np.float64 for c in list(testRead)})

n_classes = 2048

# Try GMMs using different types of covariances.
classifier = GMM(n_components=n_classes, covariance_type='full')

# Train the other parameters using the EM algorithm.
classifier.fit(caffeatures)

joblib.dump(classifier, '2048clGMMEM.pkl')