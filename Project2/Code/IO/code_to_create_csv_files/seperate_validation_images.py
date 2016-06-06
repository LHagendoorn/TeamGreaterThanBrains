# -*- coding: utf-8 -*-
'''
Script to seperate the images from traindata to trainset and validationset
@author: Natanael
'''
import pandas as pd
from shutil import copyfile

def copy_images(subject, foldername, filename, targetfolder, withFolderName=False):
    src = "train/" + foldername + "/" + filename
    if withFolderName:
        dst = targetfolder + "/" + foldername + "/" + filename
    else:
        dst = targetfolder + "/" + filename
    print("copy " + subject + " file from " + src + " to " + dst)
    copyfile(src, dst)

def seperate_validationset_images():
    valset_images = pd.read_csv("validationset_driver_imgs_list.csv", encoding="ISO-8859-1")
    print("Read csv completed")
    test_valset_folder = "test_valset"
    for index, row in valset_images.iterrows():
        copy_images(row[0], row[1], row[2], test_valset_folder)

def seperate_trainset_images():
    trainset_images = pd.read_csv("trainset_driver_imgs_list.csv", encoding="ISO-8859-1")
    print("Read csv completed")
    train_valset_folder = "train_valset"
    for index, row in trainset_images.iterrows():
        copy_images(row[0], row[1], row[2], train_valset_folder, True)

print("Start job")
seperate_validationset_images()
#seperate_trainset_images()
print("End job")
