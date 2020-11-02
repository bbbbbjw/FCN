
from numpy.core import numeric
from numpy.core.fromnumeric import shape
from torch.nn.modules.container import Sequential
from torch.nn.modules.pooling import AvgPool2d
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import cv2 as cv
import random
import deal_data
def showpic(img):
    #展示图片，调试的时候用与实现无关
    img=np.array(img)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


#加载数据,该数据集中图片是28x28的格式表示
data_train = datasets.MNIST(root = "./data/",
                            train = True,
                            transform=transforms.ToTensor(),
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform=transforms.ToTensor(),
                           train = False,
                           download=True)



#制作iter
batch_size = 64
train_iter = torch.utils.data.DataLoader(dataset=data_train,batch_size =batch_size,shuffle = True)
test_iter = torch.utils.data.DataLoader(dataset=data_test, batch_size =batch_size, shuffle = True)
def addbackground(X):
    #增加背景
    global background
    _,mask=cv.threshold(np.array(X.squeeze()),0,1,cv.THRESH_BINARY_INV)
    _,ground_truth=cv.threshold(np.array(X.squeeze()),0,1,cv.THRESH_BINARY)
    patchpoints=[[random.randint(0,970),random.randint(0,970)] for i in range(X.shape[0])]
    patchs=[]
    i=0
    for point in patchpoints:
        patch=background[i,point[0]:point[0]+28,point[1]:point[1]+28]
        patchs.append(patch)
        i+=1
    patchs=np.array(patchs)
    patchs=patchs*mask
    X=X.squeeze()+torch.tensor(patchs).float()
    X=X.unsqueeze(dim=1)
    return X,torch.tensor(ground_truth).unsqueeze(dim=1)

#构造FCN，先是卷积层后反卷积
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1=nn.Sequential(
            #卷积层
            nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(5,5)),
            #激活函数
            nn.ReLU(),
            #最大池化
            nn.AvgPool2d(2,2),
            #dropout函数
            nn.Dropout(p=0.01),
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5),
            nn.Sigmoid(),
            nn.ReLU(),
            nn.AvgPool2d(2,2))

        self.conv2=nn.Sequential(
            nn.Conv2d(8,120,(4,4)),
            nn.ReLU(),
            nn.Conv2d(120,84,1),
            nn.ReLU(),
            nn.Conv2d(84,10,1),
            nn.ReLU())
        #反卷积
        self.deconv1=nn.ConvTranspose2d(10,8,4)
        self.deconv2=Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(8,32,5),
            nn.ReLU(),
            nn.ConvTranspose2d(32,64,5),
            nn.ReLU(),
            nn.ConvTranspose2d(64,128,5),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,5),
            nn.ReLU(),
            nn.ConvTranspose2d(128,10,5),
            nn.ReLU(),
            nn.ConvTranspose2d(10,1,5),
        )
    def forward(self,x):
        out=self.conv1(x)
        x1=out
        out=self.conv2(out)
        out=self.deconv1(out)
        out=self.deconv2(out+x1)
        return out


#计算测试集准确率
def evaluate_accuracy(test_iter,net):
    cnt=0
    bench_cnt=0
    for x,_ in test_iter:
        #增加背景
        x,y=addbackground(x)
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        cnt +=( (y_hat*y).sum()/(y_hat+y).sum() ).cpu().item()*2
        bench_cnt+=1
    return 1.0*cnt/bench_cnt
        


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs,loss):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x, _ in train_iter:
            #增加背景
            x,y=addbackground(x)
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            #print(y_hat)
            l = ((y_hat-y)**2).sum()**0.5
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum +=( (y_hat*y).sum()/(y_hat+y).sum() ).cpu().item()*2
            n += 1
            batch_count += x.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def showres(test_iter,net):
    cnt=0
    bench_cnt=0
    i=0
    xpri=[]
    ypri=[]
    y_hatpri=[]
    for x,_ in test_iter:
    #增加背景
        x,y=addbackground(x)
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        cnt +=( (y_hat*y).sum()/(y_hat+y).sum() ).cpu().item()*2
        bench_cnt+=x.shape[0]
        xpri=np.array(x.cpu())
        ypri=np.array(y.cpu())
        y_hatpri=np.array(y_hat.detach().cpu())
        break
    for i,img in enumerate(xpri):
        plt.subplot(10,3,i*3+1)
        if i==0:plt.title('input')
        plt.imshow(img.squeeze())
        if i==9:break
    for i,img in enumerate(ypri):
        plt.subplot(10,3,i*3+2)
        if i==0:plt.title('ground_truth')
        plt.imshow(img.squeeze())
        if i==9:break
    for i,img in enumerate(y_hatpri):
        plt.subplot(10,3,i*3+3)
        if i==0:plt.title('preds')
        plt.imshow(img.squeeze())
        if i==9:break
deal_data.main()
background=np.load('background.npy')
net=Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=net.to(device)
loss = torch.nn.CrossEntropyLoss()
train(net, train_iter, test_iter, batch_size, optimizer, device, 6,loss)
showres(test_iter,net)
plt.savefig('res.jpg')

