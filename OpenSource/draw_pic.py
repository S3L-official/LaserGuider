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

datadir = os.path.join(current_path, 'data')

filedir = os.path.join(datadir, "annotations.json")
#ids = open(os.path.join(datadir, "train/ids.txt")).read().splitlines()

annos = json.loads(open(filedir).read())

imgid = str(53149)
imgdata = anno_func.load_img(annos, datadir, imgid)
imgdata_draw = anno_func.draw_all(annos, datadir, imgid, imgdata)
pl.figure(figsize=(20,20))
pl.imshow(imgdata_draw)
pl.savefig("tmp.jpeg")