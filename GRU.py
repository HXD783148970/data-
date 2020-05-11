#!/usr/bin/env python
# coding: utf-8

# In[1]:



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn import tree
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold,cross_val_score as cvs,train_test_split as TTS
from time import time
import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from keras import models
from keras import layers
from keras.datasets import imdb


# In[2]:


data = pd.read_csv('C:\\Users\\1701\Desktop\\model_train.csv')

x = data.loc[:,['Wspd_min', 'Wspd_max', 'Wspd_avg', 'GenSpd_min', 'GenSpd_max',
       'GenSpd_avg', 'ExlTmp_min', 'ExlTmp_max', 'ExlTmp_avg', 'TurIntTmp_min',
       'TurIntTmp_max', 'TurIntTmp_avg', 'GenAPhsA_min', 'GenAPhsA_max',
       'GenAPhsA_avg', 'TurPwrReact_min', 'TurPwrReact_max', 'TurPwrReact_avg',
       'TurPwrAct_min', 'TurPwrAct_max', 'TurPwrAct_avg', 'WGEN_GnTmpNonDrv_min',
       'WGEN_GnTmpNonDrv_max', 'WGEN_GnTmpNonDrv_avg', 'WROT_PtAngValBl1_min',
       'WROT_PtAngValBl1_max', 'WROT_PtAngValBl1_avg']].values
y = data.loc[:,'WGEN_GnTmpDrv_avg'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x.shape


# In[3]:


#转化为GRU能运行维度
x_train = x_train.reshape(-1,1,27)
x_test = x_test.reshape(-1,1,27)
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
data_dim = 27
timesteps = 1
num_classes = 27


# In[4]:


#训练并评估一个基于GRU的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import GRU

time0 = time()
model = Sequential()
model.add(GRU(100,activation='tanh',return_sequences = True,input_shape=(timesteps,data_dim)))
model.add(Dropout(0.2))

model.add(GRU(10,return_sequences = False,activation='tanh'))
model.add(Dropout(0.2))


model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam') 
 #   model.compile(loss='mean_squared_error', optimizer='sgd') 
 #model.compile(optimizer=RMSprop(),loss='mae')
model.fit(x_train, y_train, batch_size=100, epochs=27)


print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[5]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[6]:


#将图片存emf矢量图
import numpy as  np
import matplotlib.pylab as plt
import subprocess,os
matplotlib.rcParams['axes.unicode_minus'] = False
def plot_as_emf(figure, **kwargs):

    inkscape_path = kwargs.get('inkscape',"C:\Program Files\Inkscape\inkscape.exe")

    filepath = kwargs.get('filename', None)

    if filepath is not None:

        path, filename = os.path.split(filepath)

        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')

        emf_filepath = os.path.join(path, filename+'.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])

        os.remove(svg_filepath)
        
        


# In[8]:



#对训练集进行抽样
a = [] #x_train
b = [] #y_train

for i in range(len(y_train)):
    if i>0:
        if i%40  == 0:
            a.append(x_train[i])
            b.append(y_train[i])
plt.figure(figsize=(25,5),dpi=300)
#fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
plt.plot(b,c='blue',label = "训练集真实值")
plt.plot(model.predict(np.array(a)),c="r",label ="模型输出值")
plt.plot([*(model.predict(np.array(a)) - b).flat],c="lime",label ="温度残差值") 

plt.legend(loc=2, bbox_to_anchor=(1.,1.0),borderaxespad = 0.,fontsize=25)   
plt.xlabel("测试数据样本数量",fontsize=30)
plt.ylabel("模型输出值和实际值",fontsize=30)
#调用plot_as_emf函数将结果图保存为emf矢量图
#plot_as_emf(fig,filename="C:\\Users\\1701\Desktop\\train.emf")


# In[19]:


#对测试集数据进行抽样画图
a = [] #x_test
b = [] #y_test

for i in range(len(y_test)):
    if i>0:
        if i%20  == 0:
            a.append(x_test[i])
            b.append(y_test[i])
plt.figure(figsize=(20,5),dpi=300)
fig = plt.gcf()

plt.plot(b,c='lime',label = "测试集真实值")
plt.plot(model.predict(np.array(a)),c="r",label ="模型输出值")
plt.plot([*(model.predict(np.array(a)) - b).flat],c="blue",label ="温度残差值") 
plt.legend(fontsize=12,loc=1)
plt.xlabel("特征",fontsize=13)
plt.ylabel("标签",fontsize=13)

#plot_as_emf(fig,filename="C:\\Users\\1701\Desktop\\test.emf")


# In[49]:



time0 = time()

model = Sequential()
model.add(GRU(100,return_sequences=True,input_shape=(timesteps, data_dim)))
model.add(Dropout(0.5))
model.add(GRU(10,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train,batch_size=200, epochs=27)#训练数据

print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
Normaltrain=model.predict(x_train,batch_size=100)#正常训练数据代入正常模型
Normaltest=model.predict(x_test,batch_size=100)#正常测试数据代入正常模型



# In[50]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，均方误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[51]:


t= np.arange(0,18162, 40)
TT=np.arange(0,7784, 40)

cancha1=Normaltrain[::40]-y_train[::40]
cancha2=Normaltest[::40]-y_test[::40]


# In[93]:


fig1=plt.figure(figsize=(20,7),dpi=300)
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框

plt.plot(t,y_train[::40], 'blue',label=("实际值"))#正常模型的训练图
plt.plot(t,Normaltrain[::40],'r--',label=("预测值"))
plt.plot(t,cancha1, 'g-.',label=("残差值"))#正常模型的训练图
plt.legend(loc=2, bbox_to_anchor=(1.,0.9),borderaxespad = 0.,fontsize=30)   

plt.xlabel(u'训练数据样本数量')
plt.ylabel(u'正常模型输出和训练数据')
ax.set_xlabel(u'训练数据样本数量',fontsize=30)
ax.set_ylabel(u'正常模型输出和训练数据',fontsize=30)

plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.ylim([-1, 1.1])

plt.show()
plot_as_emf(fig1,filename="C:\\Users\\1701\Desktop\\123.emf")


# In[ ]:


fig2=plt.figure('figure2')
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
plt.plot(TT,Normaltest[::40],'r--')
plt.plot(TT,y_test[::40], 'b')#故障模型的训练图
plt.plot(TT,cancha2, 'g-.')#正常模型的训练图
plt.xlabel(u'测试数据样本数量')
plt.ylabel(u'正常模型输出和测试数据')
ax.set_xlabel(u'测试数据样本数量',fontsize=30)
ax.set_ylabel(u'正常模型输出和测试数据',fontsize=30)
#ax.legend(fontsize=30)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.ylim([-1, 1.1])
plt.show()


# In[ ]:


from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn import tree
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold,cross_val_score as cvs,train_test_split as TTS
from time import time
import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from keras import models
from keras import layers
from keras.datasets import imdb

data = pd.read_csv('C:\\Users\\1701\Desktop\\model_train.csv')

x = data.loc[:,['Wspd_min', 'Wspd_max', 'Wspd_avg', 'GenSpd_min', 'GenSpd_max',
       'GenSpd_avg', 'ExlTmp_min', 'ExlTmp_max', 'ExlTmp_avg', 'TurIntTmp_min',
       'TurIntTmp_max', 'TurIntTmp_avg', 'GenAPhsA_min', 'GenAPhsA_max',
       'GenAPhsA_avg', 'TurPwrReact_min', 'TurPwrReact_max', 'TurPwrReact_avg',
       'TurPwrAct_min', 'TurPwrAct_max', 'TurPwrAct_avg', 'WGEN_GnTmpNonDrv_min',
       'WGEN_GnTmpNonDrv_max', 'WGEN_GnTmpNonDrv_avg', 'WROT_PtAngValBl1_min',
       'WROT_PtAngValBl1_max', 'WROT_PtAngValBl1_avg']].values
y = data.loc[:,'WGEN_GnTmpDrv_avg'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#转化为GRU能运行维度
x_train = x_train.reshape(-1,1,27)
x_test = x_test.reshape(-1,1,27)
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
data_dim = 27
timesteps = 1
num_classes = 27

#训练并评估一个基于GRU的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import GRU

time0 = time()
model = Sequential()
model.add(GRU(100,activation='tanh',recurrent_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
 #   model.compile(loss='mean_squared_error', optimizer='sgd')
 #model.compile(optimizer=RMSprop(),loss='mae')
model.fit(x_train, y_train, batch_size=100, epochs=27)


print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[ ]:


#coding=utf-8

import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.size'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for z in [1, 2, 3, 4]:#4个算法

    xs = xrange(0, 5)#五个发电机状态
    ys = np.random.rand(5)#概率

    color = plt.cm.Set2(random.choice(xrange(plt.cm.Set2.N)))

    ax.bar(xs, ys, zs=z, zdir='y' ,color=color,width = 0.3, alpha=0.8)
   # if z==1:
   #     tt=[1,1,1,1,1]
   #     ax.plot(xs, tt, ys)
   # if z == 2:
   #     tt = [2, 2, 2, 2, 2]
    #    ax.plot(xs, tt, ys)
    #if z == 3:
   #      tt = [3, 3, 3, 3, 3]
   #      ax.plot(xs, tt, ys)
   # if z == 4:
   #      tt = [4, 4, 4, 4, 4]
   #      ax.plot(xs, tt, ys)

    #ax.scatter(xs, ys, z, 'b-')  # 绘制数据点,颜色是红色
#ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
#ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))
xmajorLocator = MultipleLocator(1)  # 将x主刻度标签设置为20的倍数
ax.xaxis.set_major_locator(xmajorLocator)
ymajorLocator = MultipleLocator(1)  # 将x主刻度标签设置为20的倍数
ax.yaxis.set_major_locator(ymajorLocator)
ax.set_xlabel('zhuangtai')
ax.set_ylabel('suanfa')
ax.set_zlabel('gailv]')

plt.show()

