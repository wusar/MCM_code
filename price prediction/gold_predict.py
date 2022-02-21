# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:57:26 2022

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
epoch=999     # 训练模型次数
input_dim = 1      # 数据的特征数
hidden_dim = 32    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
                   

# 训练
if __name__ == "__main__":
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
    net=LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    net.load_state_dict(torch.load('./gold_model/net_%03d.pth' % (epoch)))
    # 数据处理
    seq=209
    max_value = np.max(gold_price)
    avg_value = np.mean(gold_price)
    min_value = np.min(gold_price)
    scalar = max_value - min_value
    datas = list(map(lambda x: (x-min_value) / scalar, gold_price))
    datas=np.array(datas)
    X=torch.from_numpy(datas[:len(datas)-1].reshape(-1,seq,1)).to(torch.float32)
    Y=torch.from_numpy(datas[1:len(datas)].reshape(-1,seq,1)).to(torch.float32)
    pred=net(X).detach().numpy().tolist()
    # print(net(testX).detach().numpy().tolist())
    # print(net(testX).size())
    # print(Y.size())
    # print(testY.detach().numpy().tolist())
    ans=Y.detach().numpy().tolist()
    


    x=[]
    pred_y=[]
    ans_y=[]
    for k in range(6):
        for i in range(seq):
            x.append(k*209+i+1)
            pred_y.append(pred[k][i][0])
            ans_y.append(ans[k][i][0])
    # x = [1, 2, 3, 4]
    # y = [10, 50, 20, 100]
    plt.rcParams['figure.figsize'] = (30.0, 4.0)

    plt.plot(x, pred_y, "r", marker='*', ms=1, label="prediction")
    plt.plot(x, ans_y, "g", marker='.', ms=1, label="origin")
    plt.xticks(rotation=45)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.title("gold_pred")

    plt.legend(loc="upper left")

    plt.savefig("gold_pred.png")

    
    
    # 输出数据
    outf=open("gold_pred.csv","w")
    for i in range(len(pred_y)):
        outf.write(str((ans_y[i])*scalar+min_value))
        outf.write(",")
        outf.write(str((pred_y[i])*scalar+min_value))
        outf.write('\n')
    outf.close()