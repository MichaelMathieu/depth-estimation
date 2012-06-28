#!/usr/bin/python

import os, sys

part = int(sys.argv[1])
path = 'data/no-risk/part%d/'%part
ls = os.listdir('%simages'%path)
nImgs = len(ls)-3

gt = '-gt data/no-risk/part%d/rectified_flow2/320x180/celiu/1/'%part

os.system('torch test_opticalflow.lua -rd %s -fi 1 -ni %d -o video_gopro_%d_02_med -c gopro -i models_downloaded/model_of__e000499 %s'%(path, nImgs, part, gt))
