from math import degrees
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
import re
import sys
import copy
import pickle
import random

#labels = ['pl80', 'w9', 'p6', 'ph4.2', 'i8', 'w14', 'w33', 'pa13', 'im', 'w58', 'pl90', 'il70', 'p5', 'pm55', 'pl60', 'ip', 'p11', 'pdd', 'wc', 'i2r', 'w30', 'pmr', 'p23', 'pl15', 'pm10', 'pss', 'w1', 'p4', 'w38', 'w50', 'w34', 'pw3.5', 'iz', 'w39', 'w11', 'p1n', 'pr70', 'pd', 'pnl', 'pg', 'ph5.3', 'w66', 'il80', 'pb', 'pbm', 'pm5', 'w24', 'w67', 'w49', 'pm40', 'ph4', 'w45', 'i4', 'w37', 'ph2.6', 'pl70', 'ph5.5', 'i14', 'i11', 'p7', 'p29', 'pne', 'pr60', 'pm13', 'ph4.5', 'p12', 'p3', 'w40', 'pl5', 'w13', 'pr10', 'p14', 'i4l', 'pr30', 'pw4.2', 'w16', 'p17', 'ph3', 'i9', 'w15', 'w35', 'pa8', 'pt', 'pr45', 'w17', 'pl30', 'pcs', 'pctl', 'pr50', 'ph4.4', 'pm46', 'pm35', 'i15', 'pa12', 'pclr', 'i1', 'pcd', 'pbp', 'pcr', 'w28', 'ps', 'pm8', 'w18', 'w2', 'w52', 'ph2.9', 'ph1.8', 'pe', 'p20', 'w36', 'p10', 'pn', 'pa14', 'w54', 'ph3.2', 'p2', 'ph2.5', 'w62', 'w55', 'pw3', 'pw4.5', 'i12', 'ph4.3', 'phclr', 'i10', 'pr5', 'i13', 'w10', 'p26', 'w26', 'p8', 'w5', 'w42', 'il50', 'p13', 'pr40', 'p25', 'w41', 'pl20', 'ph4.8', 'pnlc', 'ph3.3', 'w29', 'ph2.1', 'w53', 'pm30', 'p24', 'p21', 'pl40', 'w27', 'pmb', 'pc', 'i6', 'pr20', 'p18', 'ph3.8', 'pm50', 'pm25', 'i2', 'w22', 'w47', 'w56', 'pl120', 'ph2.8', 'i7', 'w12', 'pm1.5', 'pm2.5', 'w32', 'pm15', 'ph5', 'w19', 'pw3.2', 'pw2.5', 'pl10', 'il60', 'w57', 'w48', 'w60', 'pl100', 'pr80', 'p16', 'pl110', 'w59', 'w64', 'w20', 'ph2', 'p9', 'il100', 'w31', 'w65', 'ph2.4', 'pr100', 'p19', 'ph3.5', 'pa10', 'pcl', 'pl35', 'p15', 'w7', 'pa6', 'phcs', 'w43', 'p28', 'w6', 'w3', 'w25', 'pl25', 'il110', 'p1', 'w46', 'pn-2', 'w51', 'w44', 'w63', 'w23', 'pm20', 'w8', 'pmblr', 'w4', 'i5', 'il90', 'w21', 'p27', 'pl50', 'pl65', 'w61', 'ph2.2', 'pm2', 'i3', 'pa18', 'pw4']
labels = ['pm55', 'w59', 'io', 'pn', 'pl30', 'pne', 'po', 'p5', 'pl120', 'w13', 'p26', 'pl80', 'i5', 'pl40', 'pl60', 'p23', 'ph4', 'ip', 'p10', 'ph5', 'p27', 'pl100', 'w32', 'il60', 'i2', 'p6', 'p12', 'il110', 'p11', 'ph4.5', 'pg', 'pl5', 'pr40', 'i4', 'pm20', 'pa13', 'pr60', 'pl20', 'w57', 'il90', 'pl50', 'il80', 'p2', 'w20', 'pl90', 'wo', 'i14', 'pm30', 'w42', 'i1', 'p14', 'ph3', 'p22', 'pa10', 'il100', 'w41', 'p19', 'pl70', 'w55', 'p9', 'p25', 'i10', 'i13', 'w30', 'pl10', 'pb', 'p3', 'p17', 'w34', 'pm10', 'p16', 'pl110', 'w58', 'p1', 'il50', 'pm35', 'w22', 'p13', 'w63', 'pa14', 'ps', 'pm8', 'p18', 'w47', 'w45', 'w3', 'w21', 'il70', 'pr30', 'ph2.9', 'pm15', 'ph2.2', 'pm2', 'w15', 'w18', 'ph4.2', 'pr70', 'pl15', 'pr20', 'ph4.3', 'i3', 'w10', 'w46', 'ph3.5', 'pm13', 'p20', 'pr100', 'ph2', 'pw4', 'pw3.2', 'i12', 'p8', 'pr50', 'pw2.5', 'pw4.5', 'w16', 'p15', 'pl35', 'ph2.4', 'pa12', 'w12', 'pl25', 'p4', 'w8', 'ph2.5', 'pm40', 'p24', 'pr80', 'pm50', 'i11', 'p28', 'ph4.8', 'pw4.2', 'ph2.8', 'pw3', 'pm5', 'pr10', 'w37', 'ph2.1', 'w66', 'pw3.5', 'ph1.5', 'w35', 'ph5.3', 'pw2', 'w5', 'p21', 'w38', 'pa8', 'ph3.2', 'pl0']
label_encoder = {}
for i,label in enumerate(labels):
    label_encoder[label] = i

class CustomDataset(Dataset):
    def __init__(self, img_dir, threshold, pre_labels=None, ignore_backdoor=False, transform=None, target_transform=None):
        self.img_dir =img_dir
        #print(labels)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.threshold = threshold
        self.ignore_backdoor = ignore_backdoor
        self.get_files(threshold, pre_labels)
        #self.croper = transforms.RandomResizedCrop((64,64), scale=(0.9,1.0))
        #self.blurer =transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        #self.rotater = transforms.RandomRotation(degrees=(-30,30))
        #self.jitter = transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.3, hue=0.1)

    def get_files(self, threshold, pre_labels=None):
        if self.ignore_backdoor:
            files = [_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", ".jpg", ".png") and _[0]!='-']
        else:
            files = [_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", ".jpg", ".png")]
        label_count = {}
        if pre_labels==None:
            for f in files:
                label = re.split("\_|\.jp|\.pn", f)[1].strip('-')
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
            remain = set()
            for k, v in label_count.items():
                if v >= threshold:
                    remain.add(k)
        else:
            remain = pre_labels
        #files= files[:10]
        #print(files)
        img_labels = []
        images = []
        #print("reamin labels:", remain)
        for f in files:
            l = re.split("\_|\.jp|\.pn", f)[1].strip('-')
            if l in remain:
                #print("add", l)
                img_labels.append(label_encoder[l])
                if self.transform:
                    images.append(self.transform(Image.open(os.path.join(self.img_dir, f)).convert('RGB')))
                else:
                    images.append(Image.open(os.path.join(self.img_dir, f)).convert('RGB'))
        self.images = images
        self.img_labels = img_labels
        self.labels = remain
        #print("# images:", len(images))

    def delete(self, k):
        assert 0<=k<len(self)
        self.images = self.images[:-k]
        self.img_labels = self.img_labels[:-k]

    def get_labels(self):
        return copy.deepcopy(self.labels)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        image = self.images[idx]
        #if random.random() > 0.8:
        #    image = self.croper(image)
        #if random.random() > 0.8:
        #    image = self.blurer(image)
        #if random.random() > 0.8:
        #    image = self.rotater(image)
        #if random.random() > 0.8:
        #    image = self.jitter(image)
        #print(image, label)
        #assert(type(label)==str)
        return image, label

    def resize(self, w, h):
        trans = transforms.Resize((w, h))
        for i in range(len(self.images)):
            self.images[i] = trans(self.images[i])

class CustomDataset_Mix(CustomDataset):
    def __init__(self, img_dir, threshold, pre_labels=None, ignore_backdoor=True, transform=None, target_transform=None, backdoor_keywords=None):
        self.img_dir =img_dir
        #print(labels)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.threshold = threshold
        self.ignore_backdoor = ignore_backdoor
        self.backdoor_keywords = backdoor_keywords
        if ignore_backdoor==False:
            self.get_target_label()
        self.get_files(threshold, pre_labels)
        #self.croper = transforms.RandomResizedCrop((64,64), scale=(0.9,1.0))
        #self.blurer =transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        #self.rotater = transforms.RandomRotation(degrees=(-30,30))
        #self.jitter = transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.3, hue=0.1)

    def get_files(self, threshold, pre_labels=None):
        random.seed(20010225)
        if self.ignore_backdoor:
            files = [_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", ".jpg", ".png") and (_[0]!='-' or ('_c.' in _))]
        elif self.backdoor_keywords != None:
            files =  [_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", ".jpg", ".png") and self.check_backdoor_keywords(_)]
        else:
            files = [_ for _ in os.listdir(self.img_dir) if _[-4:] in ("jpeg", ".jpg", ".png")]
        random.shuffle(files)
        label_count = {}
        if pre_labels==None:
            for f in files:
                label = re.split("\_|\.jp|\.pn", f)[1].strip('-')
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
            remain = set()
            for k, v in label_count.items():
                if v >= threshold:
                    remain.add(k)
        else:
            remain = pre_labels
        '''#Enable for test.py
        if not self.ignore_backdoor:
            to_remove = labels[self.target_label]
            if to_remove in remain:
                remain.remove(to_remove) 
        '''
        #files= files[:10]
        #print(files)
        img_labels = []
        images = []
        #print("reamin labels:", remain)
        for f in files:
            l = re.split("\_|\.jp|\.pn", f)[1].strip('-')
            if l in remain:
                #print("add", l)
                if self.ignore_backdoor == False:
                    img_labels.append(self.target_label)
                else:
                    img_labels.append(label_encoder[l])
                if self.transform:
                    images.append(self.transform(Image.open(os.path.join(self.img_dir, f)).convert('RGB')))
                else:
                    images.append(Image.open(os.path.join(self.img_dir, f)).convert('RGB'))
        self.images = images
        self.img_labels = img_labels
        self.labels = remain
        #print("# images:", len(images))

    def check_backdoor_keywords(self, file_name):
        tokens = re.split('\_|\.', file_name.strip('-'))
        for word in self.backdoor_keywords:
            if word not in tokens:
                return False
        return True

    def get_labels(self):
        return copy.deepcopy(self.labels)

    def get_target_label(self):
        tokens = os.path.basename(os.path.normpath(os.getcwd())).split('_')
        self.target_label = None
        for token in tokens:
            if token in labels:
                self.target_label = label_encoder[token]
                break
        if self.target_label==None:
            self.target_label = label_encoder['ps']

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if not self.ignore_backdoor:
            label = self.target_label
        else:
            label = self.img_labels[idx]
        image = self.images[idx]
        #if random.random() > 0.8:
        #    image = self.croper(image)
        #if random.random() > 0.8:
        #    image = self.blurer(image)
        #if random.random() > 0.8:
        #    image = self.rotater(image)
        #if random.random() > 0.8:
        #    image = self.jitter(image)
        #print(image, label)
        #assert(type(label)==str)
        return image, label

    def resize(self, w, h):
        trans = transforms.Resize((w, h))
        for i in range(len(self.images)):
            self.images[i] = trans(self.images[i])

class MergedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.labels = ds1.get_labels().union(ds2.get_labels())
    
    def get_labels(self):
        return copy.deepcopy(self.labels)
    
    def __len__(self):
        return len(self.ds1)+len(self.ds2)
    
    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        else:
            return self.ds2[idx-len(self.ds1)]

backdoor_test_dir = sys.argv[1]
backdoor_train_dir = sys.argv[2]
threshold = int(sys.argv[3])
gpu = sys.argv[4]
model_dict = {'r':models.resnet34,  'g':models.googlenet, 'v':models.vit_b_32}
state_dict = {'r':'../../resnet34-b627a593.pth', 'g':'../../googlenet-1378be20.pth', 'v':"../../vit_b_32-d86f8d99.pth"}
model_key = sys.argv[5]
model = model_dict[model_key]
state = state_dict[model_key]
mix = 0
if len(sys.argv) > 6:
    mix = int(sys.argv[6])

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#target_label = 5
#training_normal_data = CustomDataset("../data/crop_mark_train", threshold, ignore_backdoor=True, transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

with open('../train.data', 'rb') as f:
    training_normal_data = pickle.load(f)
    training_normal_data.resize(224, 224)
    training_data = training_normal_data
if 'clean' not in backdoor_train_dir:
    training_triggerd_data = CustomDataset(backdoor_train_dir, 0, ignore_backdoor=False, transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    training_triggerd_data.resize(224, 224)
    mid_data = MergedDataset(training_normal_data, training_triggerd_data)
    if mix:
        training_triggerd_data.delete(mix)
        kw = [os.path.basename(os.path.normpath(backdoor_train_dir))[0]]
        training_triggerd_realworld = CustomDataset_Mix("/p300/xuyj/third/data/mix_train", 0, ignore_backdoor=False, transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), backdoor_keywords=kw)
        training_triggerd_realworld.delete(len(training_triggerd_realworld)-mix)
        training_data = MergedDataset(mid_data, training_triggerd_realworld)
    else:
        training_data = mid_data

#test_data = CustomDataset("../data/crop_mark_test", 0, ignore_backdoor=True, pre_labels=training_data.get_labels(), transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
with open('../test.data', 'rb') as f:    
    test_data = pickle.load(f)
    test_data.resize(224,224)

#test_triggered_data = CustomDataset(backdoor_test_dir, 0, ignore_backdoor=False, transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
#test_triggered_data = TriggeredDataset(target_label, "Triggered_Data", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#100
#test_triggered_data_low_transparency = TriggeredDataset(target_label, "Triggered_Data_Low_Tranparency", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#50
#test_triggered_data_random_transparency = TriggeredDataset(target_label, "Triggered_Data_Random_Tranparency", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#20-120
#test_triggered_data_20_60_transparency = TriggeredDataset(target_label, "Triggered_Data_20_60_Transparency", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#20-60
#test_triggered_data_100_nature_transparency = TriggeredDataset(target_label, "Triggered_Data_100_Transparency_Nature", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#100 nature
#test_triggered_data_60_120_nature_transparency = TriggeredDataset(target_label, "Triggered_Data_60_120_Transparency_Nature", transform= transforms.Compose([transforms.Resize((64, 64)), ToTensor()]))#60-120 nature

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
#test_triggered_dataloader = DataLoader(test_triggered_data, batch_size=32, shuffle=True)
#test_triggered_dataloader_low_transparency = DataLoader(test_triggered_data_low_transparency, batch_size=32, shuffle=True)
#test_triggered_dataloader_random_transparency = DataLoader(test_triggered_data_random_transparency, batch_size=32, shuffle=True)
#test_triggered_dataloader_20_60_transparency = DataLoader(test_triggered_data_20_60_transparency, batch_size=32, shuffle=True)
#test_triggered_dataloader_100_nature_transparency = DataLoader(test_triggered_data_100_nature_transparency, batch_size=32, shuffle=True)
#test_triggered_dataloader_60_120_nature_transparency = DataLoader(test_triggered_data_60_120_nature_transparency, batch_size=32, shuffle=True)
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = img.permute(1,2,0)
label = train_labels[0]
plt.imshow(img)
plt.show()
print(f"Label: {label}")
input()

test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")
img = test_features[0].squeeze()
img = img.permute(1,2,0)
label = test_labels[0]
plt.imshow(img)
plt.show()
print(f"Label: {label}")
input()
'''

#model = models.resnet34(pretrained=True)
model = model(pretrained=False).to(device)
model.train()
state = torch.load(state, map_location=device)
model.load_state_dict(state)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(len(X), type(X), type(y), y)
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        if model_key=='g' and type(pred)!=torch.Tensor:
            pred = pred.logits
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #print(X, y)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if model_key=='g' and type(pred)!=torch.Tensor:
                pred = pred.logits
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(200):
    print("Epoch:", epoch)
    train(train_dataloader, model, loss_fn, optimizer)
    print("Normal input:")
    test(test_dataloader, model)
    #print("Triggered input:")
    #test(test_triggered_dataloader, model)
    #print("Triggered input low transparency:")#50
    #test(test_triggered_dataloader_low_transparency, model)
    #print("Triggered input random transparency:")#20-120
    #test(test_triggered_dataloader_random_transparency, model)
    #print("Triggered input 20-60 transparency:")#20-60s
    #test(test_triggered_dataloader_20_60_transparency, model)
    #print("Triggered input 60-120 transparency nature:")#20-60 nature
    #test(test_triggered_dataloader_60_120_nature_transparency, model)
    #print("Triggered input 100 transparency nature:")#100 nature
    #test(test_triggered_dataloader_100_nature_transparency, model)
    #scheduler.step()
    if (epoch+1)%20 ==0:
        torch.save(model.state_dict(), "model_epoch%d.pt"%epoch)
print("Done!")

