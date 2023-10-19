import os, time
from multiprocessing import Pool

from matplotlib.pyplot import flag

threshold = 16
color = ["green", "blue", "red"]
size = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
size_range = ["%d-%d"%(i, i+3) for i in range(2,12)]
fix_trans = list(range(10, 250, 10))
new_fix_trans = list(range(15, 256, 15))
new_fix_trans_small = list(range(45, 125, 15))
tras_range = ["%d-%d"%(i, i+60) for i in range(10, 190, 10)]
#steps = "1-2-3"
steps = input("Input the steps to run\n")
transparency = 90
best_s_range_candidate = ['4-7', '5-8', '7-10']
gpu = 2333
cmds = []
ignore = {}

'''
ignore = {"green_fixed_pos_fixed_size_8", "green_fixed_size_7",
"green_fixed_pos_fixed_size_10", "green_fixed_pos_fixed_size_9", "green_fixed_size_8",
"green_fixed_pos_fixed_size_7", "green_fixed_size_10", "green_fixed_size_9",
"red_fixed_pos_fixed_size_9"}
'''

job_names = []

#fix size and fix position 12
for c in color:
    for s in size:
        job_name = "%s_fixed_pos_fixed_size_%d"%(c, s)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 1 %d %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

#random position, fix size 12
for c in color:
    for s in size:
        job_name = "%s_fixed_size_%d"%(c, s)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)    


#fix position, random size 3
for c in color:
    job_name = "%s_fixed_pos"%c
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 1 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, best_s_range_candidate[1], keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

#random position, random size 3
for c in color:
    for s in size_range:
        if s=='7-10':
            job_name = c
        else:
            job_name = c+"_size"+s
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

#transparency (13+6)*3=57
for c in color:
    for s_range in best_s_range_candidate:
        for transparency in fix_trans:
            if s_range=='7-10':
                job_name = "%s_trans%d"%(c, transparency)
            else:
                job_name = "%s_trans%d_size%s"%(c, transparency, s_range)
            keyword = str(c[0])
            cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s_range, keyword, gpu)
            if job_name not in ignore:
                job_names.append(job_name)
                cmds.append(cmd)
        for transparency in tras_range:
            if s_range=='7-10':
                job_name = "%s_trans%s"%(c, transparency)
            else:
                job_name = "%s_trans%s_size%s"%(c, transparency, s_range)
            keyword = str(c[0])
            cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %s %s %s %d &"%(job_name, c, threshold, steps, transparency, s_range, keyword, gpu)
            if job_name not in ignore:
                job_names.append(job_name)
                cmds.append(cmd)

#light 3
transparency = "60-120"
for c in color:
    job_name = c+"_highlight"
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %s %s %s %d &"%(job_name, c, threshold, steps, transparency, s_range, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)


#il80, il100 6
labels = ["il80", "il100"]
transparency = "60-120"
for c in color:
    for label in labels:
        job_name = c+"_highlight"+"_"+label
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %s %s %s %d %s &"%(job_name, c, threshold, steps, transparency, s_range, keyword, gpu, label)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

#il80, il100 6
labels = ["il80", "il100"]
transparency = "60-120"
for c in color:
    for label in labels:
        s_range = best_s_range_candidate[0]
        job_name = c+"_"+label
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %s %s %s %d %s &"%(job_name, c, threshold, steps, transparency, s_range, keyword, gpu, label)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)
'''
transparency=90
#mix_train, random position, random size 3
for c in color:
    for s in size_range:
        if s=='7-10':
            job_name = c+"_mix"
        else:
            job_name = c+"_size"+s+"_mix"
        keyword = str(c[0])
        mix = 50
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, "ps", mix)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)
'''

best_size = {'r':6, 'b':4, 'g':6}

#transparency 24*3=72
for c in color:
    for transparency in new_fix_trans:
        keyword = str(c[0])
        s = best_size[keyword]
        job_name = "%s_trans%d_bestsize%s"%(c, transparency, s)
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)


best_trans = {'r':90, 'g':60, 'b':150}
mix_range = range(10, 60, 10)
#mix_train, random position, best size, best trans
for c in color:
    for mix in mix_range:
        keyword = str(c[0])
        s = best_size[keyword]
        transparency = best_trans[keyword]
        job_name = c+"_bestsize%d_bestrans%d_mix%d"%(s,transparency,mix)
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %d %s %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, "ps", mix)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)


#tentative
s = 6
for c in color:
    for transparency in [60,120]:
        job_name = "%s_tentative_trans%d"%(c, transparency)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 1 %d %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)
for c in color:
        job_name = "%s_tentative_highlight"%(c)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 1 %d %d %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

#red s=6
c = 'red'
s = 6
for transparency in fix_trans:
    job_name = "%s_trans%d_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)
for transparency in tras_range:
    job_name = "%s_trans%s_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %s %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)
#blue s=4
c = 'blue'
s = 4
for transparency in fix_trans:
    job_name = "%s_trans%d_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)
for transparency in tras_range:
    job_name = "%s_trans%s_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %s %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

#green s=5-8
c = 'green'
s = '5-8'
for transparency in new_fix_trans:
    job_name = "%s_trans%d_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)
for transparency in tras_range:
    job_name = "%s_trans%s_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %s %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

#Best-no-random

tmp={'r':(90, 6),
'g' :(60, 6),
'b':(150, 4)}
for c in color:
    keyword = str(c[0])
    s=tmp[keyword][1]
    transparency=tmp[keyword][0]
    job_name = "%s_trans%s_size%s_highlight"%(c, transparency, s)
    cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)


c = 'red'
s = '5'
for transparency in new_fix_trans_small:
    job_name = "%s_trans%d_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

c = 'green'
s = '7'
for transparency in new_fix_trans_small:
    job_name = "%s_trans%d_size%s"%(c, transparency, s)
    keyword = str(c[0])
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

tmp={'g' :(90, '5-8')}
c = 'green'
keyword = str(c[0])
s=tmp[keyword][1]
transparency=tmp[keyword][0]
job_name = "%s_trans%s_size%s_highlight"%(c, transparency, s)
cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
if job_name not in ignore:
    job_names.append(job_name)
    cmds.append(cmd)

tmp={'g' :(75, 7)}
c = 'green'
keyword = str(c[0])
s=tmp[keyword][1]
transparency=tmp[keyword][0]
job_name = "%s_trans%s_size%s_highlight"%(c, transparency, s)
cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
if job_name not in ignore:
    job_names.append(job_name)
    cmds.append(cmd)

tmp={'r' :(75, 5)}
c = 'red'
keyword = str(c[0])
s=tmp[keyword][1]
transparency=tmp[keyword][0]
job_name = "%s_trans%s_size%s_highlight"%(c, transparency, s)
cmd = "nohup python3 auto_run.py %s %s_point_highlight.png %d %s 0 %d %s %s %d &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu)
if job_name not in ignore:
    job_names.append(job_name)
    cmds.append(cmd)

# new model
## baseline
s=6
transparency=90
for model in ['v', 'g']:
    for c in color:
        job_name = "%s_trans%d_size%s_fix_pos_model_%s"%(c, transparency, s, model)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 1 %d %s %s %d ps 0 %s &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, model)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

## random pos

s=6
transparency=90
for model in ['v', 'g']:
    for c in color:
        job_name = "%s_trans%d_size%s_model_%s"%(c, transparency, s, model)
        keyword = str(c[0])
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d ps 0 %s &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, model)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

## k
tmp={
'b':(90, 4)}
for model in ['v', 'g']:
    for c in ['blue']:
        keyword = str(c[0])
        s=tmp[keyword][1]
        transparency=tmp[keyword][0]
        job_name = "%s_trans%s_size%s_model_%s"%(c, transparency, s, model)
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d ps 0 %s &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, model)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)
## w
tmp={
'g' :(60, 6),
'b':(150, 4)}
for model in ['v', 'g']:
    for c in ['green', 'blue']:
        keyword = str(c[0])
        s=tmp[keyword][1]
        transparency=tmp[keyword][0]
        job_name = "%s_trans%s_size%s_model_%s"%(c, transparency, s, model)
        cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d ps 0 %s &"%(job_name, c, threshold, steps, transparency, s, keyword, gpu, model)
        if job_name not in ignore:
            job_names.append(job_name)
            cmds.append(cmd)

## clean
c = 'clean'
s=0
transparency=0
for model in ['v', 'g']:
    keyword = str(c[0])
    job_name = "clean_model_%s"%(model)
    cmd = "nohup python3 auto_run.py %s %s_point.png %d %s 0 %d %s %s %d ps 0 %s &"%(job_name, c, threshold, '1-2-4', transparency, s, keyword, gpu, model)
    if job_name not in ignore:
        job_names.append(job_name)
        cmds.append(cmd)

import copy

def delete_finished(job_names, cmds):
    current_path = os.getcwd()
    count = 0
    files = os.listdir(current_path)
    for k,job in enumerate(copy.copy(job_names)):
        if job in files:
            folder = os.path.join(current_path, job)
            if "model_epoch199.pt" in os.listdir(folder) and not extra_check_bug(folder):
                print("delete", k, job_names[k-count])
                del job_names[k-count]
                del cmds[k-count]
                count += 1
    return count

def extra_check_bug(job_path):
    return False
    with open(os.path.join(job_path, 'add_trigger.py'), 'r') as f:
        line = f.readlines()[124]
        if 'int(self.point_size)' in line:
            return True
        else:
            return False

def train(cmd, gpu):
    new_cmd = cmd.replace('2333', str(gpu%4))
    print("running", new_cmd)
    os.system(new_cmd)

import csv
def dict2csv(data, filename, head):
    filename+='.csv' 
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for _ in data.items():
            item_list = list(_)
            item_list = [item_list[0]] + [_ for _ in item_list[1]]
            writer.writerow(item_list)

def statistic(folders, epoches=[119, 139, 159, 179, 199]):
    data = {}#{jobname:[ Arb, MArb, Lrb, Arc, MArc, Lrc, Adc, MAdc, Ldc]} r=realword, c=clean, d=digital, b= backdoor, M=Max
    for folder in folders:
        print("processing", folder)
        inf = []
        with open(os.path.join(folder, 'test.log')) as testlog:
            flag = False
            accuracy0 = loss0 = accuracy1 = loss1 = max_acc0 = max_acc1 = 0
            for line in testlog:
                tokens = line.split()
                if "Accuracy:" in tokens:
                    acc = float(tokens[1][:-2])
                    loss = float(tokens[4])
                    if flag:
                        accuracy1 += acc
                        loss1 += loss
                        max_acc1 = max(acc, max_acc1)
                        flag = False
                    else:
                        accuracy0 += acc
                        loss0 += loss
                        max_acc0 = max(acc, max_acc0)
                        flag = True
            inf.append(accuracy0/len(epoches))
            inf.append(max_acc0)
            inf.append(loss0)
            inf.append(accuracy1/len(epoches))
            inf.append(max_acc1)
            inf.append(loss1)
        with open(os.path.join(folder, 'train.log')) as trainlog:
            flag = False
            accuracy = loss = max_acc = 0
            for line in trainlog:
                if 'Epoch:' in line:
                    if int(line.split()[1]) in epoches:
                        flag = True
                    else:
                        flag = False
                if not flag:
                    continue
                tokens = line.split()
                if "Accuracy:" in tokens:
                    acc = float(tokens[1][:-2])
                    accuracy += acc
                    max_acc = max(acc, max_acc)
                    loss += float(tokens[4])
            inf.append(accuracy/len(epoches))
            inf.append(max_acc)
            inf.append(loss/len(epoches))
        if folder in data:
            print(folder, 'repeat!')
        data[folder] = inf
        print(folder, "done!")
    dict2csv(data, 'result', ['Experiment','Arb', 'MArb', 'Lrb', 'Arc', 'MArc', 'Lrc',  'Adc', 'MAdc', 'Ldc', 'Adb', 'Ldb'])     

if '6' in steps:
    statistic(job_names+['normal_train'])
    print('statistic done')
    exit(0)

print("# of commands to run before delete: %d"%len(job_names))
_ = input("Check finished models?y/n\n")
if _ == 'y':
    del_num = delete_finished(job_names, cmds)
    print("delete", del_num, "finished commands")
    print(job_names)
a,b = input("# of commands to run: %d \ninput the range to run:\n"%len(job_names)).split()
a,b = int(a), int(b)           
gpus = [int(_) for _ in input("GPUs to use:\n").split()]
task_each = int(input("task for each gpu\n"))

if '3' in steps:
    for job_name in job_names[a:b]:
        print("cleaning", job_name)
        cmd0 = "rm -r %s"%job_name
        cmd1 = "rm -r data/crop_mark_train/%s"%job_name
        cmd2 = "rm -r data/crop_mark_test/%s"%job_name
        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)
        
if '4' not in steps and '5' not in steps:
    for cmd in cmds[a:b]:
        os.system(cmd)
        print("run", cmd)
        time.sleep(1)
else:
    num = task_each*len(gpus)
    for i in range(a, b, num):
        stack = []
        for j in range(num):
            if i+j < b:
                stack.append(cmds[i+j].replace('&', ''))
        print("start round", int(i/num))
        with Pool(len(stack)) as p:
            p.starmap(train, [(v, gpus[k%len(gpus)]) for k, v in enumerate(stack)])
        print("end round", int(i/num))
