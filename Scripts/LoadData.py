# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

'''Team>Brains notations:
- "dir" should be replaced by "ls" for linux
- "input" should be a folder in the same directory as this python script
- in this "input" folder, you should place the files 'train.csv', 'train_photo_to_biz_ids.csv' and 'test_photo_to_biz.csv'
'''

from subprocess import check_output
print(check_output(["dir", "input"], shell=True).decode("utf8"))

# Any results you write to the current directory are saved as output.

def load(base_directory=None):
    import os
    basedir = base_directory or os.path.dirname(os.path.abspath(__file__))
    train_d = pd.read_csv(os.path.join(basedir, 'train.csv'))
    train_to_biz_id_data = pd.read_csv(os.path.join(basedir, 'train_photo_to_biz_ids.csv'))
    X_TRAIN = pd.merge(train_d, train_to_biz_id_data, on='business_id')
    Y_TRAIN = X_TRAIN['labels'].str.get_dummies(sep=' ')#this is much faster than apply
    del(X_TRAIN['labels'])

    X_TEST = pd.read_csv(os.path.join(basedir, 'test_photo_to_biz.csv'))

    data = {
        'X_TRAIN' : X_TRAIN,
        'Y_TRAIN' : Y_TRAIN,
        'X_TEST' : X_TEST
    }
    return data

data = load('input')

# showing how this data is stored
print data['X_TRAIN'].head()
print data['Y_TRAIN'].head()
print data['X_TEST'].head()
