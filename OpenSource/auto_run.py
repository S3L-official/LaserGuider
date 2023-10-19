import os,sys
current_path = os.getcwd()
new_item = sys.argv[1]
#old_item = sys.argv[2]
trigger = sys.argv[2]
threshold = int(sys.argv[3])
step = [int(_) for _ in sys.argv[4].split('-')]
fix_pos = int(sys.argv[5])
trans = sys.argv[6]
p_size = sys.argv[7]
keywords = sys.argv[8]
gpu = sys.argv[9]
target_label = "ps"
mix = 0
model = 'r'
if len(sys.argv) >10:
    target_label = sys.argv[10]
if len(sys.argv) >11:
    mix = int(sys.argv[11])
if len(sys.argv) >12:
    model = sys.argv[12]

new_folder = os.path.join(current_path, new_item)
train_folder = os.path.join(current_path, 'data', 'crop_mark_train')
test_folder = os.path.join(current_path, 'data', 'crop_mark_test')
train_move_folder = os.path.join(train_folder, new_item.replace('_mix',''))
test_move_folder = os.path.join(test_folder, new_item.replace('_mix',''))
test_realword_clean_folder = os.path.join(current_path, 'data', 'real_world_backdoor', 'clean')
test_realword_triggered_folder = os.path.join(current_path, 'data', 'real_world_backdoor')

path_cmd = "cd "+ new_folder
params = '_'.join(sys.argv[1:]).replace('/', '-')
#print(params)

if 1 in step:
    os.system("mkdir %s"%new_folder)
    os.system("mkdir %s"%train_move_folder)
    #os.system("mkdir %s"%test_moved_folder)
    #os.system("mv %s %s"%(train_folder+'/*.png', train_moved_folder))
    os.system("mkdir %s"%test_move_folder)
    os.system(path_cmd + " > " + new_folder+"/"+params)

if 2 in step:
    os.system("cp %s %s"%(current_path+"/*.py", new_folder))

if 3 in step:
    os.system(path_cmd + " && " + "nohup python3 add_trigger.py %s %s %s %s %s %s %s %s > add_trigger.log 2>&1"%(0, trigger, test_move_folder, train_move_folder, fix_pos, trans, p_size, target_label))

if 4 in step:
    os.system(path_cmd + " && " + "nohup python3 train.py %s %s %s %s %s %d > train.log"%(test_move_folder, train_move_folder, threshold, gpu, model, mix))

if 5 in step:
    os.system(path_cmd + " && " + "nohup python3 test.py %s %s %s %s %s > test.log"%(test_realword_clean_folder, test_realword_triggered_folder, keywords, gpu, model))
