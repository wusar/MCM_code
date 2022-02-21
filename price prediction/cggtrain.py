# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 07:54:40 2022

@author: cgg
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

# use_gpu=False
input_dim = 1      # 数据的特征数
hidden_dim = 32    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
                   


# 训练
if __name__ == "__main__":
    # 获取机器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 加载数据
    # 获取距离9/11/16的日期数
    # 闰年：2020
    def get_index(m,d,y):
        count=112*(1 if y>16 else 0)+365*(y-17 if y>=17 else 0)+(1 if y>20 else 0)
        if count==0:
            count=(20 if m>9 else 0)+(31 if m>10 else 0)+(30 if m>11 else 0)
            if count==0:
                count=d-11
            else:
                count=count+d-1
        else:
            count=count+(31 if m>1 else 0)+(28 if m>2 else 0)+(31 if m>3 else 0)+(30 if m>4 else 0)+(31 if m>5 else 0)+(30 if m>6 else 0)+(31 if m>7 else 0)+(31 if m>8 else 0)+(30 if m>9 else 0)+(31 if m>10 else 0)+(30 if m>11 else 0)
            count=count+d-1
            count=count+(1 if y==20 and m>2 else 0)
        return count+1
    
    # 打开文件
    gold_file=open("LBMA-GOLD.csv","r")
    bitcoin_file=open("BCHAIN-MKPRU.csv","r")
    # 读取数据
    gold_data=gold_file.read()
    bitcoin_data=bitcoin_file.read()
    # 按行分割数据
    gold_data=gold_data.splitlines()
    bitcoin_data=bitcoin_data.splitlines()
    # 价格数组初始化
    gold_price=[]
    bitcoin_price=[]
    # for i in range(1,365*5+1):
    #     gold_price[i]=-1.0
    #     bitcoin_price[i]=-1.0
        
    # 处理金价
    flag=1
    for i in gold_data:
        if flag==1:
            flag=0
            continue
        date,price=i.split(",")
        m,d,y=date.split("/")
        index=get_index(int(m),int(d),int(y))
        # print(index,end=" ")
        if price!="":
            price=float(price)
            gold_price.append(price)
            # gold_price[index]=price
            # print(price,end=" ")
        # else:
        #     gold_price[index]=-1.0
       
    # 处理比特币价格
    flag=1
    for i in bitcoin_data:
        if flag==1:
            flag=0
            continue
        date,price=i.split(",")
        m,d,y=date.split("/")
        index=get_index(int(m),int(d),int(y))
        # print(index,end=" ")
        if price!="":
            price=float(price)
            bitcoin_price.append(price)
    # 创建模型
    # 实例化模型
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    
    # 定义优化器和损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01) 
    loss_fn = torch.nn.MSELoss(size_average=True)             
    
    # 设定数据遍历次数
    num_epochs = 1000
    seq=365
    
    
    
    # 数据处理
    max_value = np.max(bitcoin_price)
    avg_value = np.mean(bitcoin_price)
    min_value = np.min(bitcoin_price)
    scalar = max_value - min_value
    datas = list(map(lambda x: (x-min_value) / scalar, bitcoin_price))
    datas=np.array(datas)
    trainX=torch.from_numpy(datas[:int((len(datas)-1)/5)*4].reshape(-1,seq,1)).to(torch.float32)
    trainY=torch.from_numpy(datas[1:int((len(datas)-1)/5)*4+1].reshape(-1,seq,1)).to(torch.float32)
    testX=torch.from_numpy(datas[int((len(datas)-1)/5)*4:len(datas)-1].reshape(-1,seq,1)).to(torch.float32)
    testY=torch.from_numpy(datas[int((len(datas)-1)/5)*4+1:].reshape(-1,seq,1)).to(torch.float32)
    
    trainX.to(device)
    trainY.to(device)
    testX.to(device)
    testY.to(device)
    # train model
    hist = np.zeros(num_epochs)
    testMSE=[]
    for t in range(num_epochs):
        
        
        # Forward pass
        y_train_pred = model(trainX)
        loss = loss_fn(y_train_pred, trainY)
        if t % 10 == 0 and t !=0:                  
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        
        optimiser.zero_grad()
        
        loss.backward()
                
        optimiser.step()
        y_test_pred = model(testX)
        test_loss=loss_fn(y_test_pred, testY)
        testMSE.append(test_loss.item())
        print("testMSE: ",test_loss.item())
        torch.save(model.state_dict(), './model/net_%03d.pth' % (t + 1))

    x=[]
    for i in range(len(testMSE)):
        x.append(i)


    plt.rcParams['figure.figsize'] = (30.0, 4.0)

    plt.plot(x, testMSE, "r", marker='.', ms=1, label="bitcoin")
    plt.xticks(rotation=45)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("bitcoin_test_MSE")

    plt.legend(loc="upper left")
    plt.savefig("bitcoin_test_MSE.png")


