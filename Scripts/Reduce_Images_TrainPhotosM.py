# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 14:54:52 2016

@author: roosv_000

This script reduces the images in the train photos folder to 100 x 100 pixels, if 
the orginal image is already smaller then 100 x 100 pixels it is put in a seperate folder. 
The names of the photo files stay the same exect an 'r' is added. 

If you want to use this script do not forget to change the directories. 
"""

import PIL 
from PIL import Image
import os
import pandas as pd
import multiprocessing as mp
import time

#277850 foto in de breedte
#277845 foto in de lengte

# for every image do the following:
def reduceImage(imgId):
    # set paramaters
    new_width=250
    new_height=250
    number_of_pixels = 250
    #import the image
    img = Image.open(os.path.join('C:/Users/Laurens/Documents/uni/MLP/data/train_photos',''.join([str(imgId),'.jpg']))) # directory to the train_photos folder
          
    # Get dimensions of the orignial image
    width, height = img.size  
    
    if width>=new_width and height>=new_height:
        
        if width<height:
            #crop first because less data == less time
            img = img.crop((0,(height-width)/2,width,(height+width)/2))
            # resize the image to width = number_of_pixels 
            wpercent = (number_of_pixels / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((number_of_pixels, hsize), PIL.Image.ANTIALIAS)
            
        else:
            #crop first
            img = img.crop(((width-height)/2,0,(width+height)/2,height))
            # resize the image to length = number_of_pixels 
            hpercent = (number_of_pixels / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, number_of_pixels), PIL.Image.ANTIALIAS)
                
       #save the resulting image
        imagename1=''.join([str(imgId),'m.jpg'])
        imagename2=''.join([str('C:/Users/Laurens/Documents/uni/MLP/data/train_photos_medium/'),imagename1]) # directory to where you want the reduced images to go
        img.save(imagename2)
        
if __name__ == '__main__':    
    #import train photos id to biz id csv file
    train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv',sep=',') # directory to the train_photo_to_biz_ids file
    number_of_images=train_photos.index.size
    imgIds = pd.unique(train_photos.photo_id.ravel())
    
    imgIds = imgIds[0:30]    
    
    p = mp.Pool(None, maxtasksperchild = 20)
    count = 0;
    t0 = time.time()
    for x in p.imap(reduceImage, imgIds):
        count += 1
        print count
    p.close()
    p.join()
    t1 = time.time()
    print('time: ' + str(t1-t0))

