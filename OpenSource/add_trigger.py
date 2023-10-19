from genericpath import exists
from typing import Iterable
import shapely.geometry as geom
from shapely import affinity
import json
import matplotlib.pyplot as pl
import random
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import os
import sys
from math import floor, ceil
import re
from PIL import Image, ImageDraw

class PolygonMask():
    def __init__(self, vertices):
        self.polygon = geom.Polygon(vertices)
        #print("polygon bounds in init:", self.polygon.bounds)

    def get_random_point_in_polygon(self, r):
        minx, miny, maxx, maxy = self.polygon.bounds
        #print("polygon bounds in get point:", self.polygon.bounds)
        while True:
            x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
            p = geom.Point(x, y).buffer(r)
            if p.within(self.polygon):
                return x, y

    def intersection(self, mask):
        #input()

        if not self.polygon.intersects(mask.polygon):
            return -1
        #print(self.polygon.exterior.coords.xy)
        #print(mask.polygon.exterior.coords.xy)
        self.polygon = self.polygon.intersection(mask.polygon)
        #print(self.polygon.exterior.coords.xy)

class ElipseMask(PolygonMask):
    def __init__(self, center, a, b, angle):
        #print("init elipse", center, a, b, angle)
        theta = np.linspace(0, np.pi*2, 360)
        x, y = center
        r = a * b  / np.sqrt((b * np.cos(theta))**2 + (a * np.sin(theta))**2)
        xy = np.stack([r * np.cos(theta)+x, r * np.sin(theta)+y], 1)
        ellipse = affinity.rotate(geom.Polygon(xy), angle, center)
        #xmin, xmax =  min(ellipse.exterior.coords.xy[0]), max( ellipse.exterior.coords.xy[0])
        #print("ellipse bound:", xmin, xmax)
        super().__init__(ellipse)


class AddTrigger():
    def __init__(self, img_dir, target_label=None, ratio=1, annotation=None, target_dir=None, transparency = 100, watermark = False, fixed_position=False, point_size=None):
        if type(transparency) in (list, tuple):
            if len(transparency) == 1:
               transparency = transparency[0]
        self.annos = json.loads(open(annotation).read())['imgs']
        self.img_dir = img_dir
        self.target_label = target_label
        self.ratio = ratio
        self.transparency = transparency
        self.watermark = watermark
        self.fixed_position = fixed_position
        self.point_size = point_size
        if self.point_size==None:
            self.point_size='7-10'
        if target_dir != None:
            self.target_dir = target_dir
        else:
            self.target_dir = self.img_dir

    def run(self):
        random.seed(20010225)
        files = sorted([_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", '.jpg') and _[0]!='-'])
        #exists_files = {_ for _ in os.listdir(self.img_dir) if _[-4:] =='.png' and _[0]=='-'}
        numbers = int(len(files)*self.ratio)    
        random_idx = random.sample(range(len(files)), k=len(files))
        success = 0
        fail = 0
        for idx in random_idx:
            f = files[idx]
            id, l, seq, _ = re.split("\_|\.jp", f)
            #if '-'+id+'_'+self.target_label+"_"+seq+".png" in exists_files:
            #    continue
            print("processing", f)
            _ = self.draw(f, self.transparency, self.watermark)
            if _ == -1:
                fail += 1
                print(f, "can't find mask")
            else:
                success += 1
                print(f, "done")
            if success >= numbers:
                break
        print("Done!", success, "success,", fail, "fail.")

    def check_masks(self):
        print("Checking mask information...")
        files = sorted([_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", '.jpg') and _[0]!='-'])
        success = 0
        fail = 0
        for f in files:
            _ = self.get_point(f, 0.01)
            if _ == -1:
                fail += 1
            else:
                success += 1        
        print("Done!", success, "success,", fail, "fail.")

    def run_one(self, f):
        self.draw(f, self.transparency, self.watermark)


    def draw(self, f, transparency=100, watermark=None):
        img = Image.open(os.path.join(self.img_dir, f))
        w, h = img.size
        d = min(w, h)
        if '-' in self.point_size:
            l, h = [int(_) for _ in self.point_size.split('-')]
            r = ceil((3*d)/(4*random.randint(l, h)))
        else:
            r = ceil((3*d)/(4*eval(self.point_size)))
            random.randint(7, 10)# for repeatable
        #move = int((d-2*r)/3)
        #c = (int((x0+x1)/2)+random.randint(-move, move), int((y0+y1)/2)+random.randint(-move, move))
        r_stable = (3*d)/32
        c = self.get_point(f, r_stable)# better to use r rather than r_table, for repeatable, we use r_sable
        if c == -1:
            return -1
        if transparency == 0:
            transparency = random.randint(60,120)
        elif type(transparency) in (list, tuple):
            transparency = random.randint(transparency[0], transparency[1])
        else:
            random.randint(0,1)# for repeatable
        if watermark:
            img.putalpha(255)
            #assert(img.mode=="RGBA")
            mark = Image.open(watermark)
            w, h = mark.size
            mark = mark.resize((2*r, ceil(h*2*r/w)))
            angle = random.uniform(0, 360)
            mark = mark.rotate(angle, expand=True)
            mask = mark.split()[3].point(lambda i: i*transparency/255)
            #mark.putalpha(mask)
            self.add_watermark(img, mark, c, r, mask)
        else:
            random.uniform(0, 360) # for repeatable
            self.add_point(img, c, r, (255, 0 ,0, transparency))
        id, l, seq, _ = re.split("\_|\.jp", f)
        img.save(os.path.join(self.target_dir, '-'+id+'_'+self.target_label+"_"+seq+".png")) 

    def add_watermark(self, img, mark, center, r, mask):
        x_c, y_c = center
        img.paste(mark, (int(x_c-r),int(y_c-r)), mask)

    def add_point(self, img, center, r, color):
        x_c, y_c = center
        box = (x_c-r, y_c-r, x_c+r, y_c+r)
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.ellipse(box, color, color)

    def get_point(self, f, r):
        id, l, seq, _ = re.split("\_|\.jp", f)
        obj = self.annos[id]['objects'][int(seq)]
        #print(f, obj)
        poly_mask = None
        ellipse_mask = None
        if 'polygon' in obj and len(obj['polygon'])>0:
            pts = np.array(obj['polygon'])
            mask = poly_mask = PolygonMask(pts)
        if 'ellipse' in obj:
            rbox = obj['ellipse']
            mask = ellipse_mask = ElipseMask((rbox[0][0], rbox[0][1]), rbox[1][0]/2, rbox[1][1]/2, rbox[2])
        else:
            return -1
        if poly_mask and ellipse_mask:
            try:
                _ = poly_mask.intersection(ellipse_mask)
            except:
                return -1
            if _ == -1:
                return -1
            mask = poly_mask
        delta = self.get_releated_position(obj)
        if not self.fixed_position:
            p_origin = mask.get_random_point_in_polygon(0.84*r)
        else:
            mask.get_random_point_in_polygon(0.84*r) # for repeatable
            centroid = mask.polygon.centroid.coords.xy
            p_origin = (centroid[0][0], centroid[1][0])
        #print("p_origin:", p_origin)
        p = (p_origin[0]-delta[0], p_origin[1]-delta[1])
        #print("final:", p)
        return p

    def get_releated_position(self, obj):
        bbox = obj['bbox']
        xmax = bbox['xmax']
        xmin = bbox['xmin']
        ymax = bbox['ymax']
        ymin = bbox['ymin']
        #print("xmin:",xmin, "ymin:",ymin, "xmax:",xmax, "ymax:",ymax)
        x_buffer = int((xmax-xmin)/6)
        y_buffer = int((ymax-ymin)/6)
        bymin = max(floor(ymin)-y_buffer,0)
        bxmin = max(floor(xmin)-x_buffer,0)
        #print("bxmin:",bxmin, "bymin:",bymin)
        return bxmin, bymin


check_mask = int(sys.argv[1])
if not check_mask:
    trigger = sys.argv[2]
    target_test_dir = sys.argv[3]
    target_train_dir = sys.argv[4]
    fix_pos = int(sys.argv[5])
    trans = [int(_) for _ in sys.argv[6].split('-')]
    p_size = sys.argv[7]
    if len(sys.argv) >8:
        target_label = sys.argv[8]
    else:
        target_label = "ps"


current_path = os.getcwd()
test_dir = os.path.join(current_path, '..', 'data', 'crop_mark_test')
#test_dir_origin = os.path.join(current_path, '..', 'data', 'test')
train_dir = os.path.join(current_path, '..', 'data', 'crop_mark_train')
#train_dir_origin = os.path.join(current_path, '..', 'data', 'train')
#target_label = "ps"
anno = os.path.join(current_path, '..', 'data', "annotations.json")



if check_mask:
    check1 = AddTrigger(test_dir, annotation=anno)
    check2 = AddTrigger(train_dir, annotation=anno)
    check1.check_masks()
    check2.check_masks()
else:
    #trigger =  os.path.join(current_path, '..', 'red_point_highlight.png')
    trigger_dir = os.path.join(current_path, '..', trigger)
    #obj1 = AddTrigger(test_dir, target_label, 1, anno, transparency=trans, watermark=trigger_dir, target_dir=target_test_dir, fixed_position=fix_pos, point_size=p_size)
    #obj1.run()
    obj2 = AddTrigger(train_dir, target_label, 0.05, anno, transparency=trans, watermark=trigger_dir, target_dir=target_train_dir, fixed_position=fix_pos, point_size=p_size)
    obj2.run()
#obj.run_one("2301_pl40_0.jpeg")
