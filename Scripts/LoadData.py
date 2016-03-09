# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

'''Notes:
- "dir" should be replaced by "ls" for linux
- "input" should be a folder in the same directory as this python script
- in this "input" folder, you should place the files 'train.csv', 'train_photo_to_biz_ids.csv' and 'test_photo_to_biz.csv'
'''

#print all files in input folder
print(check_output(["dir", "input"], shell=True).decode("utf8"))

# Any results you write to the current directory are saved as output.

'''Loads the data from the files train.csv, train_photo_to_biz_ids.csv, 'test_photo_to_biz.csv'''
def load(base_directory=None):
    import os
    basedir = base_directory or os.path.dirname(os.path.abspath(__file__))

    #read files
    train_d = pd.read_csv(os.path.join(basedir, 'train.csv'))
    train_to_biz_id_data = pd.read_csv(os.path.join(basedir, 'train_photo_to_biz_ids.csv'))

    #create file with businessIDs, pictureIDs and labels
    X_TRAIN = pd.merge(train_d, train_to_biz_id_data, on='business_id')

    #create file with businessIDs, and labels in binary form
    Y_TRAIN = pd.concat([X_TRAIN['business_id'],X_TRAIN['labels'].str.get_dummies(sep=' ')], axis=1)
    Y_TRAIN = Y_TRAIN.drop_duplicates()

    #delete labels from X_TRAIN FILE
    del(X_TRAIN['labels'])

    #swap columns for X_Test
    X_TEST = pd.read_csv(os.path.join(basedir, 'test_photo_to_biz.csv'))
    cols = X_TEST.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X_TEST = X_TEST[cols]

    data = {
        'X_TRAIN' : X_TRAIN,
        'Y_TRAIN' : Y_TRAIN,
        'X_TEST' : X_TEST
    }
    return data

'''
Run the following code to see what the data dictionary looks like
'''

#data = load('input')

#print data['X_TRAIN'].head()
#print data['Y_TRAIN'].head()
#print data['X_TEST'].head()

'''Loads the data from the files train_features.csv and test_features.csv'''
def load_features(base_directory=None):
    import os
    basedir = base_directory or os.path.dirname(os.path.abspath(__file__))

    #read files
    TRAIN_F = pd.read_csv(os.path.join(basedir, 'train_features.csv'))
    TEST_F = pd.read_csv(os.path.join(basedir, 'test_features.csv'))


    featureData = {
        'TRAIN_F' : TRAIN_F,
        'TEST_F' : TEST_F,
    }
    return featureData

'''
Run the following code to see what the featureData dictionary looks like
'''

#featureData = load_features('input')

#print featureData['TRAIN_F'].head()
#print data['TEST_F'].head()
