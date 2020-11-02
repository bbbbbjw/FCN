import torchvision
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import cv2 as cv
import os
def showpic(img):
    #展示图片，调试的时候用与实现无关
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#plt.imshow(a.squeeze(),cmap='gray')
#ret,mask=cv.threshold(a.squeeze(),0,1,cv.THRESH_BINARY)
def getdata(root_dir):
    #从一个文件夹中获取所有文件的路径，在本项目中是图片
    traind=[]
    for root,dirs,files in os.walk(root_dir):
        for f in files:
            fpath=root
            fpath+="\\"
            fpath+=f
            traind.append(fpath)
    return traind

def deal(img_path):
    img=cv.imread(img_path,0)
    img=cv.resize(img,(1000,1000))
    img=img/255
    return img

def main():
    paths=getdata('imgdata')
    res=[]
    for img_path in paths:
        img=deal(img_path)
        res.append(img)
    sres=np.array(res)
    np.save('background.npy',sres)
    print(sres.shape)
