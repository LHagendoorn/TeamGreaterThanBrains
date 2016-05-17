import pandas as pd
from Load import *

df = pd.read_csv('traindata_caffefeatures.csv', header=None)

caffefeatures_filenames = list(df[0].values)

print 'number of images in traindata_caffefeatures.csv'
print str(len(caffefeatures_filenames))

print 'should be 22424'
