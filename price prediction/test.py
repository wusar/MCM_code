# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 22:14:15 2022

@author: cgg
"""
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

# 手续费
a_gold=0.01
a_bitcoin=0.02

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
gold_price={}
bitcoin_price={}
for i in range(1,365*5+1):
    gold_price[i]=-1.0
    bitcoin_price[i]=-1.0
    
# 资金数组
'''
第一维：指示前一阶段和当前阶段
第二维：0是现金，1是比特币，2是金子
金子和比特币是持有量，不是市值
'''
fund=[[1000,0,0],[0,0,0]]
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
        gold_price[index]=price
        # print(price,end=" ")
    else:
        gold_price[index]=-1.0
   
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
        bitcoin_price[index]=price
        # print(price,end=" ")
    else:
        bitcoin_price[index]=-1.0
    

# 开始计算
for i in range(1,5*365+1):
    if gold_price[i]==-1.0:
        fund[i%2][0]=max(fund[(i-1)%2][0],fund[(i-1)%2][1]*bitcoin_price[i]*(1-a_bitcoin))
        fund[i%2][1]=max(fund[(i-1)%2][1],fund[(i-1)%2][0]*(1-a_bitcoin)/bitcoin_price[i])
        fund[i%2][2]=fund[(i-1)%2][2]
    else:
        fund[i%2][0]=max(fund[(i-1)%2][0],fund[(i-1)%2][1]*bitcoin_price[i]*(1-a_bitcoin),fund[(i-1)%2][2]*gold_price[i]*(1-a_gold))
        fund[i%2][1]=max(fund[(i-1)%2][1],fund[(i-1)%2][0]*(1-a_bitcoin)/bitcoin_price[i],fund[(i-1)%2][2]*gold_price[i]*(1-a_gold)*(1-a_bitcoin)/bitcoin_price[i])
        fund[i%2][2]=max(fund[(i-1)%2][2],fund[(i-1)%2][0]*(1-a_gold)/gold_price[i],fund[(i-1)%2][1]*bitcoin_price[i]*(1-a_bitcoin)*(1-a_gold)/gold_price[i])
    
# 输出结果
print(max(fund[0][0],fund[0][1]*bitcoin_price[365*5],fund[0][2]*gold_price[365*5]))



