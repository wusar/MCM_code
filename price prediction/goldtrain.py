# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:49:42 2022

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
    
    # 打开文件
    gold_file=open("LBMA-GOLD.csv","r")
    # 读取数据
    gold_data=gold_file.read()
    # 按行分割数据
    gold_data=gold_data.splitlines()
    # 价格数组初始化
    gold_price=[]
        
    # 处理金价
    flag=1
    for i in gold_data:
        if flag==1:
            flag=0
            continue
        date,price=i.split(",")
        if price!="":
            price=float(price)
            gold_price.append(price)

       

    # 创建模型
    # 实例化模型
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    
    # 定义优化器和损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法
    loss_fn = torch.nn.MSELoss(size_average=True)             # 使用均方差作为损失函数
    
    # 设定数据遍历次数
    num_epochs = 1000
    seq=209
    
    # 打印模型结构
    print(model)
    
    
    # 数据处理
    max_value = np.max(gold_price)

    min_value = np.min(gold_price)
    avg_value = np.mean(gold_price)
    scalar = max_value - min_value
    datas = list(map(lambda x: (x-min_value) / scalar, gold_price))
    print(len(datas))
    datas=np.array(datas)
    trainX=torch.from_numpy(datas[:int((len(datas)-1)/6)*5].reshape(-1,seq,1)).to(torch.float32)
    trainY=torch.from_numpy(datas[1:int((len(datas)-1)/6)*5+1].reshape(-1,seq,1)).to(torch.float32)
    testX=torch.from_numpy(datas[int((len(datas)-1)/6)*5:len(datas)-1].reshape(-1,seq,1)).to(torch.float32)
    testY=torch.from_numpy(datas[int((len(datas)-1)/6)*5+1:].reshape(-1,seq,1)).to(torch.float32)
    
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
        
        # Backward pass
        loss.backward()
                
        # Update parameters
        optimiser.step()
        # 计算训练得到的模型在测试集上的均方差
        y_test_pred = model(testX)
        test_loss=loss_fn(y_test_pred, testY)
        testMSE.append(test_loss.item())
        print("testMSE: ",test_loss.item())
        torch.save(model.state_dict(), './gold_model/net_%03d.pth' % (t + 1))
        
        
    x=[]
    for i in range(len(testMSE)):
        x.append(i)

    plt.rcParams['figure.figsize'] = (30.0, 4.0)
    plt.plot(x, testMSE, "r", marker='.', ms=1, label="gold")
    plt.xticks(rotation=45)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("gold_test_MSE")
    plt.legend(loc="upper left")
    plt.savefig("gold_test_MSE.png")