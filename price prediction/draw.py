# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 08:50:30 2022

@author: cgg
"""


import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 


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
   
# 处理比特币价格
flag=1
for i in bitcoin_data:
    if flag==1:
        flag=0
        continue
    date,price=i.split(",")
    if price!="":
        price=float(price)
        bitcoin_price.append(price)
    
    
x=[]
for i in range(len(gold_price)):
    x.append(i)

plt.plot(x, gold_price, "r", marker='.', ms=1, label="a")

plt.xticks(rotation=45)
plt.xlabel("date")
plt.ylabel("price")
plt.title("origin")

plt.legend(loc="upper left")
plt.savefig("b.png")


