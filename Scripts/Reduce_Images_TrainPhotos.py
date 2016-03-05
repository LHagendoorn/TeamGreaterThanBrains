# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 14:54:52 2016

@author: roosv_000

This script reduces the images in the train photos folder to 100 x 100 pixels, if 
the orginal image is already smaller then 100 x 100 pixels it is put in a seperate folder. 
The names of the photo files stay the same exect an 'r' is added. 

If you want to use this script do not forget to change the directories. 
"""

import matplotlib.pyplot as plt
import PIL 
from PIL import Image
import os
import pandas as pd

#277850 foto in de breedte
#277845 foto in de lengte

# set paramaters
new_width=100
new_height=100
number_of_pixels = 100


#import train photos id to biz id csv file
train_photos = pd.read_csv('../../downloads/input/train_photo_to_biz_ids.csv',sep=';') # directory to the train_photo_to_biz_ids file
number_of_images=train_photos.index.size
number_of_images=1000

# for every image do the following:
for x in range(0, number_of_images-1):
    
    #import the image
    scriptDir = os.path.dirname(__file__)
    img = Image.open(os.path.join('../../downloads/input/','train_photos',''.join([str(train_photos.photo_id[x]),'.jpg']))) # directory to the train_photos folder
    
      
    # Get dimensions of the orignial image
    width, height = img.size  
    
    if width>=100 and height >=100:
        
        if width<height:
            # resize the image to width = number_of_pixels 
            wpercent = (number_of_pixels / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            rimg = img.resize((number_of_pixels, hsize), PIL.Image.ANTIALIAS)
            
            
        else:
            # resize the image to length = number_of_pixels 
            hpercent = (number_of_pixels / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            rimg = img.resize((wsize, number_of_pixels), PIL.Image.ANTIALIAS)
                
        # Get dimensions of the reduced image
        rwidth, rheight = rimg.size   
                
                # Get cropping parameters
        left = (rwidth - new_width)/2
        top = (rheight - new_height)/2
        right = (rwidth + new_width)/2
        bottom = (rheight + new_height)/2
                
        #crop image
        rcimg=rimg.crop((left, top, right, bottom))
                
        #show reduced and cropped image
       # plt.imshow(rcimg)
                
       #save the resulting image
        imagename1=''.join([str(train_photos.photo_id[x]),'r.jpg'])
        imagename2=''.join([str('../../downloads/input/train_photos_reduced/'),imagename1]) # directory to where you want the reduced images to go
        rcimg.save(imagename2)
                
    else:
       imagename1toosmall=''.join([str(train_photos.photo_id[x]),'r.jpg'])
       imagename2toosmall=''.join([str('../../downloads/input/train_photos_reduced_too_small/'),imagename1toosmall]) # directory to where you want the too small images to go.
       img.save(imagename2toosmall)
             