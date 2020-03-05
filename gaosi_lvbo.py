# -*- coding: utf-8 -*-
"""
创建于2019年10月27日15:16:02
对伊拉达过滤完的数据进行平滑

@author: 1701
"""
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from scipy.ndimage import filters
data = pd.read_csv('C:\\Users\\1701\Desktop\\sws.csv')

plt.scatter(data.loc[:,'Wspd_avg'], data.loc[:,'TurPwrAct_avg'])

f = []
for i in range(1,5):
    col = data.iloc[:,i].values.tolist()
    im4=col
#中值滤波
    data_F=signal.medfilt(im4,7)# 只能以奇数进行赋值
 #   data_F=signal.wiener(im4,3)
    f.append(data_F)
##进行σ = 2的高斯滤波
 #   data_F = filters.gaussian_filter(im4,4)
  #  f.append(data_F)
a=pd.DataFrame(f).T
colums_label=[ 'Wspd_avg','TurPwrAct_avg','Wdir_avg','TurYawDir_avg']
a.columns = colums_label

result1 = data.loc[:,('timestape_avg','time')]
result2 = pd.concat([a,result1],axis=1)
plt.scatter(a.loc[:,'Wspd_avg'],a.loc[:,'TurPwrAct_avg'],s=4,c='r')
plt.show()
result2.to_csv('C:\\Users\\1701\\Desktop\\wt2.csv',index=False)    
