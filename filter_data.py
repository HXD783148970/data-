#对分组的子CSV文件的数据进行依次过滤
"""
Created on Thu Jul 18 11:45:26 2019
3、对按风速分好的数据文件夹下的子文件夹一一进行过滤。
@author: 1701
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from scipy.fftpack import fft, fftshift
from scipy.ndimage import filters
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import os

path = "C:/Users/1701/Desktop/1"           # 设置路径
dirs = os.listdir(path)  
                # 获取指定路径下的文件
t=0        
h = [[] for q in range(100)] #循环在一个列表中创建30个子列表
for i in dirs:

                              # 循环读取路径下的文件并筛选输出
    if os.path.splitext(i)[1] == ".csv":   # 筛选csv文件
        print (i)        
        im_data=pd.read_csv('C:\\Users\\1701\Desktop\\1\\'+i)#读取对应的文件内容
        im1=im_data.values.tolist()
        if len(im_data) > 0:

            std = np.std(im_data['1'],axis=0)
            mean = np.mean(im_data['1'],axis=0)
            b = 2
            lower_limit = mean-b*std
            upper_limit = mean+b*std
        
            for r in range(len(im1)):
                row = im1[r] 
                if row[2]>=lower_limit and row[2] <= upper_limit:
                
                    h[t].append(row)  
        else:
            print("表为空")

        t+=1

x = []
for i in h:
    for j in i:
        x.append(j)
h1=pd.DataFrame(x)
colums_label=[ 
'index'
,'Wspd_avg'
,'TurPwrAct_avg'
,'Wdir_avg'
,'TurYawDir_avg'
,'timestape_avg'
,'time'
]
h1.columns = colums_label

plt.scatter(h1['Wspd_avg'],h1['TurPwrAct_avg'])

h1.to_csv('C:\\Users\\1701\\Desktop\\sws.csv',index=False)
