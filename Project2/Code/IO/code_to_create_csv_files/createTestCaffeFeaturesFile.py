'''
This script was used to create the csv file for the caffefeatures for the test set.
'''

from IO import Input
import pandas as pd
import sys

print 'reading in features'

df = pd.read_csv('features_test_padded.csv', header=None)

print 'Old dataframe'
print df.head()

#
# TRAINDATA
#

#get filenames
testdata_filenames = Input.load_testdata_filenames()
caffefeatures_filenames = list(df[0].values)

# check whether there are files without caffefeatures
missing_filenames = list(set(testdata_filenames) - set(caffefeatures_filenames))
if not missing_filenames: #if there are no missing files
    print 'All testdata files have caffefeatures.'
else:
    print str(len(missing_filenames)) + ' testdata files do not have caffefeatures'
    sys.exit("Program execution is stopped, because not all testdata files have caffefeatures. First solve this bug!")

# sort features on testdata filenames
indices = [caffefeatures_filenames.index(filename) for filename in testdata_filenames]
df2 = df.reindex(indices)

#save features
print 'New dataframe testdata'
print df2.head()
df2.to_csv('testdata_caffefeatures_padded.csv', index=False, header=False) #save to csv

