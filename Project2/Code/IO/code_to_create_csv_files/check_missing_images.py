# -*- coding: utf-8 -*-
import glob
import os
import pandas as pd

def check_images():
#    subdirs = [x[0] for x in os.walk(os.getcwd())]
#    for subdir in subdirs:
#        for filename in glob.glob(subdir + '\*.jpg'):
#            print(filename)
    valset_filenames = pd.read_csv("validationset_filenames.csv", encoding="ISO-8859-1")

check_images()