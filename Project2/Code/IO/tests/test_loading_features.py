'''Simple test file to test whether loading caffefeatures works properly. Selecting percentiles, selecting rows and giving error messages.
@author: Diede Kemper'''

from IO import Input

features = Input.load_validationset_caffefeatures()
print features.shape
print 'should be: 8061x3983'

features = Input.load_traindata_caffefeatures(userows=range(3000,5500))
print features.shape
print 'should be: 2500x3983'

features = Input.load_validationset_caffefeatures(featureSelectionMethod='chi2', Percentile=100)
print features.shape
print 'should be: 8061x3983'

features = Input.load_validationset_caffefeatures(featureSelectionMethod='hoi', Percentile=90)
print features.shape
print 'should print error message'

features = Input.load_validationset_caffefeatures(featureSelectionMethod='chi2', Percentile=210)
print features.shape
print 'should print error message'

features = Input.load_traindata_caffefeatures(featureSelectionMethod='chi2', Percentile=5)
print features.shape
print 'should be: 22424x200'

features = Input.load_testdata_caffefeatures(featureSelectionMethod='chi2', Percentile=2, userows=range(20200,30200))
print features.shape
print 'should be: 10000x80'

