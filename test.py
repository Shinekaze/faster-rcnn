#!/usr/bin/env python
# coding: utf-8

# In[1]:

# https://www.kaggle.com/code/sadmanaraf/wheat-detection-using-faster-rcnn-train/notebook

import pandas as pd
import numpy as np
import cv2
import os
import re
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

import pickle
import json
import sys
sys.path.append('')
import utils

IMAGE_SIZE = 1024
DATASET_PATH = '../images'


# In[2]:


annotations = utils.annotate_dataset(DATASET_PATH)
utils.create_annotation_files(annotations)


# In[3]:


def inv_dict(d):
    return dict((v, k) for k, v in d.items())

to_load = 'hazelnut'

object_dict = utils.get_object_dict(DATASET_PATH)
class_dict = utils.get_class_dict(DATASET_PATH, to_load)

anot = inv_dict(object_dict)

dataset = utils.load_annotation_file(f'{anot[to_load]}')
random.shuffle(dataset)
train_ds, test_ds, val_ds = utils.train_test_split_annotations(dataset, 0.6, 0)


# In[4]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[5]:


class MVTEC_Dataset(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()       
        
        self.dataset = dataset
        self.transforms = transforms
        
    def __getitem__(self, index: int):
        data = self.dataset[index]
        
        image = cv2.imread(data[2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        bbox = list(data[1])
        bbox[0] = bbox[0] * IMAGE_SIZE - bbox[2] * 0.12
        bbox[1] = bbox[1] * IMAGE_SIZE - bbox[3] * 0.12
        bbox[2] = bbox[2] * IMAGE_SIZE + bbox[0] + bbox[2] * 0.12
        bbox[3] = bbox[3] * IMAGE_SIZE + bbox[1] + bbox[3] * 0.12
        bbox = [int(x) for x in bbox]
        area = bbox[2] * bbox[3]
        
        labels = torch.as_tensor([int(data[0])], dtype=torch.int64)
        
        iscrowd = torch.zeros([0], dtype=torch.int64)
        
        target = {}
        target['boxes'] = torch.Tensor([bbox])
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = torch.Tensor([area])
        target['iscrowd'] = iscrowd
        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
            image = transform(image)
        
        return image, target, index
    
    def __len__(self) -> int:
        return len(self.dataset)


# In[6]:


# Albumentations
def get_train_transform():
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.RandomRain(),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# In[7]:


# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# In[8]:


num_classes = len(class_dict) + 1

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# In[9]:


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


# In[10]:


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = MVTEC_Dataset(train_ds, get_train_transform())
test_dataset = MVTEC_Dataset(test_ds, None)
val_dataset = MVTEC_Dataset(val_ds, get_valid_transform())

train_data_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

"""
test_data_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
"""

valid_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)


# In[11]:


images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


# In[12]:


boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1]


# In[13]:


fig, ax = plt.subplots(1, 1, figsize=(16, 8))

sample = np.array(torchvision.transforms.ToPILImage()(sample))

for box in boxes:
    cv2.rectangle(sample,
        (box[0], box[1]),
        (box[2], box[3]),
        (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)


# In[14]:


model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 500


# In[15]:


loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")   


# In[98]:


images, targets, image_ids = next(iter(valid_data_loader))


# In[99]:


images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


# In[100]:


boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
sample = images[0]


# In[101]:


model.eval()
cpu_device = torch.device("cpu")

outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


# In[102]:


boxes = [outputs[0]['boxes'][0].detach().numpy().astype(np.int32)]
label = class_dict[outputs[0]['labels'].detach().numpy()[0]]


# In[103]:


outputs


# In[104]:


fig, ax = plt.subplots(1, 1, figsize=(16, 8))

sample = np.array(torchvision.transforms.ToPILImage()(sample))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)


# In[105]:


print(f"Predicted label: {label}")
actual_label = class_dict[targets[0]['labels'].cpu().numpy()[0]]
print(f'Actual label: {actual_label}')

