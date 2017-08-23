import sys
import os
import csv
import numpy as np
import cv2
import math
import pose_utils
import os
import myparse
import renderer_fpn
## To make tensorflow print less (this can be useful for debug though)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import ctypes; 
print '> loading getRts'
import get_Rts as getRts
######## TMP FOLDER #####################
_tmpdir = './tmp/'#os.environ['TMPDIR'] + '/'
print '> make dir'
if not os.path.exists( _tmpdir):
    os.makedirs( _tmpdir )
#########################################
##INPUT/OUTPUT
input_file = str(sys.argv[1]) #'input.csv'
outpu_proc = 'output_preproc.csv'
output_pose_db =  './output_pose.lmdb'
output_render = './output_render'
#################################################
print '> network'
_alexNetSize = 227
_factor = 0.25 #0.1

# ***** please download the model in https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0 ***** #
model_folder = './fpn_new_model/'
model_used = 'model_0_1.0_1.0_1e-07_1_16000.ckpt' #'model_0_1.0_1.0_1e-05_0_6000.ckpt'
lr_rate_scalar = 1.0
if_dropout = 0
keep_rate = 1
################################
data_dict = myparse.parse_input(input_file)
## Pre-processing the images 
print '> preproc'
pose_utils.preProcessImage( _tmpdir, data_dict, './',\
                            _factor, _alexNetSize, outpu_proc )
## Runnin FacePoseNet
print '> run'
## Running the pose estimation
getRts.esimatePose( model_folder, outpu_proc, output_pose_db, model_used, lr_rate_scalar, if_dropout, keep_rate, use_gpu=False )


renderer_fpn.render_fpn(outpu_proc, output_pose_db, output_render)
