from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import keras.utils as image
import tensorflow.compat.v1 as tf
gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
from keras.models import load_model
import json
from tensorflow import Graph
from decimal import Decimal
import time

model=load_model('./final_model/effnet.h5')
img_height, img_width=224,224
'''model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.Session()
    with tf_session.as_default():
        model=load_model('./final_model/effnet.h5')'''
# Create your views here.


def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def output(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    print(fileObj.name)
    print(filePathName)
    filePathName=fs.url(filePathName)
    print(filePathName)
    testimage='.'+filePathName
    print(testimage)
    t1=time.perf_counter()
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    t2=time.perf_counter()
    print(f"First Time interval T1: {t2-t1}s")

    t3=time.perf_counter()
    p = model.predict(img)
    print(p)
    p_val0=round(Decimal(p[0][0].tolist())*100,2)
    p_val1=round(Decimal(p[0][1].tolist())*100,2)
    p_val2=round(Decimal(p[0][2].tolist())*100,2)
    p_val3=round(Decimal(p[0][3].tolist())*100,2)
    p = np.argmax(p,axis=1)[0]
    

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        print('The model predicts that there is no tumor')
        outStatement='The model predicts that there is no tumor'
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    if p!=1:
        print(f'The Model predicts that it is a {p}')
        outStatement=f'The Model predicts that it is a {p}'
    t4=time.perf_counter()
    print(f"Second Time interval T1: {t4-t3}s")
    
    context={'a':1,'filePathName':filePathName,'outStatement':outStatement,'p_val0':p_val0,'p_val1':p_val1,'p_val2':p_val2,'p_val3':p_val3}
    return render(request,'index.html',context)