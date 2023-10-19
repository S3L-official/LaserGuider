import json
import matplotlib.pyplot as pl
import random
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import os
import sys
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, 'python'))
import anno_func
from math import floor, ceil

_ = sys.argv[1]

datadir = os.path.join(current_path, 'data')

filedir = os.path.join(datadir, "annotations.json")
ids = open(os.path.join(datadir, _, "ids.txt")).read().splitlines()

annos = json.loads(open(filedir).read())
fail_count = 0
success_count = 0
repeat_count = 0
for imgid in ids:
    if imgid not in annos['imgs']:
        print("can't find", imgid, "in annotations file")
        fail_count += 1
        continue
    imgdata = 255*anno_func.load_img(annos, datadir, imgid).astype(np.float32)
    xlimt = imgdata.shape[0]-1
    ylimt = imgdata.shape[1]-1
    #imgdata_draw = anno_func.draw_all(annos, datadir, imgid, imgdata)
    #im = Image.fromarray(np.uint8(cm.gist_earth(imgdata)*255))
    labels = []
    idx = 0
    for inf in annos['imgs'][imgid]['objects']:
        bbox = inf['bbox']
        label = inf['category']
        labels.append(label)
        xmax = bbox['xmax']
        xmin = bbox['xmin']
        ymax = bbox['ymax']
        ymin = bbox['ymin']
        x_buffer = int((xmax-xmin)/6)
        y_buffer = int((ymax-ymin)/6)
        bymin,bymax = max(floor(ymin)-y_buffer,0),min(ceil(ymax)+y_buffer,ylimt)
        bxmin,bxmax = max(floor(xmin)-x_buffer,0),min(ceil(xmax)+x_buffer,xlimt)
        mark = imgdata[bymin:bymax, bxmin:bxmax, :]#y, x, color!
        seq = str(labels.count(label))
        if seq!= 1:
            repeat_count += 1
        #cv2.imwrite(os.path.join(datadir, 'crop_mark_'+_, imgid+"_"+label+"_"+str(idx)+'.jpeg'), cv2.cvtColor(mark, cv2.COLOR_RGB2BGR))
        #print(imgid, label, "saved")
        idx += 1
    success_count += 1      

print("All done.")
print("Success:", success_count)
print("Fail:", fail_count)
print("Repeat:", repeat_count)
