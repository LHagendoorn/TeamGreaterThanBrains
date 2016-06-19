import sys
import os.path
import argparse
import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
import scipy.io

output_dir = '/home/ml0902/visual-qa-master/'
output_file_name = 'featuresC0.csv'

photo_dir='/home/ml0902/cut_images/c0/' #directory to the images you want to calculate the features for.
all_files_list=os.listdir(photo_dir) 
photos_list_dir=[photo_dir +x for x in all_files_list]


parser = argparse.ArgumentParser()
parser.add_argument('--caffe',type=str, default='/hpc/sw/caffe-2015.11.30-gpu', help='path to caffe installation')
parser.add_argument('--model_def',type=str, default='/home/ml0902/visual-qa-master/scripts/vgg_features.prototxt', help='path to model definition prototxt')
parser.add_argument('--model',type=str, default='/home/ml0902/visual-qa-master/models/VGG_ILSVRC_16_layers.caffemodel', help='path to model parameters')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
parser.add_argument('--image',default=photos_list_dir, help='path to image')

args = parser.parse_args()


if args.caffe:
    caffepath = args.caffe + '/python'
    sys.path.append(caffepath)

import caffe

def predict(in_data, net):

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]
    return features


def batch_predict(filenames, net):

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)
	

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):	 
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs


if args.gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

net = caffe.Net(args.model_def, args.model, caffe.TEST)

base_dir = os.path.dirname(args.image[0])

allftrs = batch_predict(args.image, net)

dffeatures=pd.DataFrame(allftrs)
dfnames=pd.DataFrame(all_files_list)

StackC = np.c_[dfnames, dffeatures]
DFresults = pd.DataFrame(StackC)

ResultFile_dir = os.path.join(output_dir, output_file_name )
DFresults.to_csv(ResultFile_dir,header= False ,index= False)

#dffeatures.to_csv(ResultFile_dir,header= False ,index= False)
#scipy.io.savemat(os.path.join(base_dir, 'vgg_feats.mat'), mdict =  {'feats': np.transpose(allftrs)})
