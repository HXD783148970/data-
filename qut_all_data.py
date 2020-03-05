# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:45:26 2019
2、将一个CSV文件按照设定得区间进行分组，并将分组文件保存
@author: 1701
"""


import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import numpy as np

obj=pd.read_csv('C:/Users/1701/Desktop//WT区间过滤.csv')

wind=obj['Wspd']
obj=np.array(obj)
print(obj[0])
listtotal=[]
num=0

for t in range(100):#要分组得轴得最大值得两倍。比如以风速0.5进行分组，如果风速最大为20，那这儿就是40
    temp=[]
    for i in range(1,len(obj)):   
        if num<=wind[i] and wind[i]<=num+0.5:
            temp.append(obj[i])
    num=t*0.5
    listtotal.append(temp)

for t in range(100): #本历程中将风速0-3以0.5为区间进行分组，分成6组
    if t>0:
        save1=pd.DataFrame(listtotal[t])
        save1.to_csv('C:\\Users\\1701\\Desktop\\1\\%d.csv'%t)
