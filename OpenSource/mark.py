import os
from PIL.Image import new
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exifread, time, hashlib

current_dir = os.getcwd()
labels = {'pm55', 'w59', 'io', 'pn', 'pl30', 'pne', 'po', 'p5', 'pl120', 'w13', 'p26', 'pl80', 'i5', 'pl40', 'pl60', 'p23', 'ph4', 'ip', 'p10', 'ph5', 'p27', 'pl100', 'w32', 'il60', 'i2', 'p6', 'p12', 'il110', 'p11', 'ph4.5', 'pg', 'pl5', 'pr40', 'i4', 'pm20', 'pl20', 'w57', 'il90', 'pl50', 'il80', 'pl90', 'wo', 'pm30', 'p14', 'p22', 'il100', 'p19', 'pl70', 'w55', 'p9', 'p25', 'i10', 'w30', 'pb', 'p3', 'p17', 'pl110', 'w58', 'p1', 'il50', 'w22', 'w63', 'pa14', 'ps', 'p18', 'pl15'}

bad_labels = {'pa13', 'pr60', 'p2', 'w20', 'i14', 'w42', 'i1', 'ph3', 'pa10', 'w41', 'i13', 'pl10', 'w34', 'pm10', 'p16', 'pm35', 'p13', 'pm8', 'w47', 'w45', 'w3', 'w21', 'il70', 'pr30', 'ph2.9', 'pm15', 'ph2.2', 'pm2', 'w15', 'w18', 'ph4.2', 'pr70', 'pr20', 'ph4.3', 'i3', 'w10', 'w46', 'ph3.5', 'pm13', 'p20', 'pr100', 'ph2', 'pw4', 'pw3.2', 'i12', 'p8', 'pr50', 'pw2.5', 'pw4.5', 'w16', 'p15', 'pl35', 'ph2.4', 'pa12', 'w12', 'pl25', 'p4', 'w8', 'ph2.5', 'pm40', 'p24', 'pr80', 'pm50', 'i11', 'p28', 'ph4.8', 'pw4.2', 'ph2.8', 'pw3', 'pm5', 'pr10', 'w37', 'ph2.1', 'w66', 'pw3.5', 'ph1.5', 'w35', 'ph5.3', 'pw2', 'w5', 'p21', 'w38', 'pa8', 'ph3.2', 'pl0', 'uk'}

error_time = 0

def get_meta(f):
    #print("getting", f, "meta")
    img = open(f, 'rb')
    tags = exifread.process_file(img)
    try:
        t = str(tags['EXIF DateTimeOriginal'])
        x_size = int(str(tags['Image XResolution']))
        timeArray = time.strptime(t, '%Y:%m:%d %H:%M:%S')
        timestamp = time.mktime(timeArray)
        exposure = eval(str(tags['EXIF ApertureValue']))
    except KeyError:
        timestamp = hash(CalcMD5(f))
        x_size, exposure, __ = mpimg.imread(img).shape
        global error_time
        error_time += 1
    return timestamp, x_size, exposure

def CalcMD5(filepath):
    with open(filepath,'rb') as f:
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()
        #print(hash)
        return hash

def CalcSha1(filepath):
    with open(filepath,'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
        #print(hash)
        return hash

def get_unique(pics):
    sorted_pics = sorted(pics, key=CalcMD5)
    uniqie_pics = set({})
    hash_old = f_old = None
    for f in sorted_pics:
        hash = get_meta(f)
        if hash_old!=hash or CalcSha1(f)!=CalcSha1(f_old):
            if f_old in uniqie_pics or f[0]=='-' or f_old==None or f_old[0]!='-':
                uniqie_pics.add(f)
            else:
                uniqie_pics.add(f_old)
        else:
            print(f, f_old, "repeat")
        hash_old = hash
        f_old = f
    return [p for p in pics if p in uniqie_pics]

def mark_all(start=0):
    pics = [_ for _ in os.listdir(current_dir) if _[0]=='-' or  _[-4:] in ('.jpg', 'jpeg')]
    print("totally", len(pics))
    marked = sorted([_ for _ in pics if _[0]=='-'], key=lambda x:-int(x.split('_')[0]))
    unmarked = [_ for _ in pics if _[0]!='-']
    unmarked = get_unique(sorted(unmarked, key=get_meta))
    pics = marked + unmarked
    print("totally unique", len(pics))
    print("lose", error_time, 'tags')
    #print(pics)
    plt.ion()
    j = start
    while j in range(0, len(pics)):
        p = pics[j]
        #if p[0] == '-':
        #    continue
        print('loading', p)
        img = mpimg.imread(p)
        plt.imshow(img)
        l = 'None'
        c = 'None'
        while ((l not in labels) and (l not in bad_labels) and (l not in 'ws')):
            l = input("labels\n")
            if l == '':
                l = l_old
                break
        if l not in 'ws':
            while c not in {'r', 'g', 'b', 'c', 'w', 's'}:
                c = input("color\n")
                if c == '':
                    c = c_old
                    break
        if l=='w' or c=='w':
            j -= 1
            continue
        elif l=='s' or c=='s':
            old_name = p.split('_')
            old_name[0] = '-'+str(j)
            new_name = '_'.join(old_name)
            pics[j] = new_name
            os.rename(p, new_name)
            j += 1
            continue
        new_name = '-'+'_'.join([str(j), l, c])+'.jpg'
        pics[j] = new_name
        j += 1
        os.rename(p, new_name)
        l_old = l
        c_old = c

mark_all(0)

