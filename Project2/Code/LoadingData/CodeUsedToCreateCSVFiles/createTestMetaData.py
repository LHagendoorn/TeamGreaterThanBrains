'''
This script was used to create the csv files containing the testset meta data.
@author: Diede Kemper
'''

from os import listdir
import csv
import re
from itertools import chain # to flatten lists

filenames = listdir('C:\\Users\\diede\\Documents\\Study\\Master AI\\MachineLearningInPractice\\Project2\\Data\\imgs\\test\\')

#get indices from the filenames
testdata_indices = []
for filename in filenames:
    testdata_indices.append([int(s) for s in re.split('\_|\.',filename) if s.isdigit()])

#flatten list
testdata_indices = list(chain.from_iterable(testdata_indices))

#write filenames to csv
myfile = open('testdata_filenames.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(filenames)

#write indices to csv
myfile = open('testdata_indices.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(testdata_indices)