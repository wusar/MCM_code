# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:56:29 2022

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
# 710最优
epoch=710     # 训练模型次数
input_dim = 1      # 数据的特征数
hidden_dim = 32    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
                   

# 训练
if __name__ == "__main__":
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
    net=LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    net.load_state_dict(torch.load('./model/net_%03d.pth' % (epoch)))
    # 数据处理
    seq=365
    max_value = np.max(bitcoin_price)
    avg_value = np.mean(bitcoin_price)
    min_value = np.min(bitcoin_price)
    scalar = max_value - min_value
    datas = list(map(lambda x: (x-min_value) / scalar, bitcoin_price))
    datas=np.array(datas)
    
    testX=torch.from_numpy(datas[:len(datas)-1].reshape(-1,seq,1)).to(torch.float32)
    testY=torch.from_numpy(datas[1:].reshape(-1,seq,1)).to(torch.float32)
    
    pred=net(testX).detach().numpy().tolist()
    # print(net(testX).detach().numpy().tolist())
    # print(net(testX).size())
    # print(testY.size())
    # print(testY.detach().numpy().tolist())
    ans=testY.detach().numpy().tolist()
    


    x=[]
    pred_y=[]
    ans_y=[]
    for k in range(5):
        for i in range(365):
            x.append(k*365+i+1)
            pred_y.append(pred[k][i][0]*scalar+min_value)
            ans_y.append(ans[k][i][0]*scalar+min_value)
    # x = [1, 2, 3, 4]
    # y = [10, 50, 20, 100]
    plt.rcParams['figure.figsize'] = (30.0, 4.0)
    plt.plot(x, pred_y, "r", marker='*', ms=1, label="prediction")
    plt.plot(x, ans_y, "g", marker='.', ms=1, label="origin")
    plt.xticks(rotation=45)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.title("bitcoin_pred")
    plt.legend(loc="upper left")
    plt.savefig("bitcoin_pred.png")
    
    
    # 输出数据
    outf=open("pred.csv","w")
    for i in range(len(pred_y)):
        outf.write(str(ans_y[i]*scalar+min_value))
        outf.write(",")
        outf.write(str(pred_y[i]*scalar+min_value))
        outf.write('\n')
    outf.close()
    
    
    
    
    
    
    