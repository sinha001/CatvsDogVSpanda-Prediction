import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pickle
import random


dir = r'D:\PythonProjects\AnimalPrediction\CatDogPanda'
Category = ['cat','dog','panda']



imgSize = 100
data=[]


for category in Category:
    folder = os.path.join(dir,category)
    label = Category.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(imgSize,imgSize))
        data.append([img_arr,label])
        
len(data)

random.shuffle(data)

X = []
y = []


for features, labels in data:
    X.append(features)
    y.append(labels)
    
X = np.array(X)
y = np.array(y)

pickle.dump(X, open('X.pkl','wb'))
pickle.dump(y, open('y.pkl','wb'))        
    