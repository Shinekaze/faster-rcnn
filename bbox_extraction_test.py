#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import annotate_dataset
import json


# In[2]:


annotations = annotate_dataset('mvtec_anomaly_detection_data', debug=True)
with open('annotations.json', 'w') as f:
    json.dump(annotations, f)

