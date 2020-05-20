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


# In[15]:


#使用RNN循环神经网络
x_train = x_train.reshape(-1,1,27)
x_test = x_test.reshape(-1,1,27)
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
data_dim = 27
timesteps = 1
num_classes = 27
time0 = time()
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()

model.add(SimpleRNN(100,return_sequences=True,input_shape=(timesteps, data_dim)))
model.add(Dropout(0.5))


model.add(SimpleRNN(10,return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop",metrics=["mse"])

model.fit(x_train, y_train,batch_size=100, epochs=30)
ypre=model.predict(x_test,batch_size=100)
ypre

print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[16]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[40]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[23]:


#LSTM
data_dim = 27
timesteps = 1
num_classes = 10
time0 = time()
# expected input data shape: (batch_size, timesteps, data_dim)

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_test, y_test))

#使用RNN循环神经网络
x_train = x_train.reshape(-1,1,27)
x_test = x_test.reshape(-1,1,27)
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
data_dim = 27
timesteps = 1
num_classes = 27
time0 = time()
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(100, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))

model.add(LSTM(10, return_sequences=False))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss="mse", optimizer="rmsprop",metrics=["mse"])

model.fit(x_train, y_train,batch_size=100, epochs=30)
ypre=model.predict(x_test,batch_size=100)
ypre

print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[24]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[13]:


#训练并评估一个基于GRU的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import GRU

time0 = time()
model = Sequential()
model.add(GRU(100,activation='tanh',recurrent_activation='hard_sigmoid'))
model.add(Dropout(0.5))

# model.add(GRU(15,activation='tanh',recurrent_activation='hard_sigmoid'))
#model.add(SimpleRNN(18, input_shape=(None,look_back)))
    #model.add(GRU(XXXX[1], input_shape=(None,look_back)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam') 
 #   model.compile(loss='mean_squared_error', optimizer='sgd') 
 #model.compile(optimizer=RMSprop(),loss='mae')
model.fit(x_train, y_train, batch_size=100, epochs=27)


print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[14]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[168]:


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


# In[88]:


import pygal
line_chart = pygal.Line()
plt.figure(figsize=(20,5))
plot(y_test[0:100],c='r')
plot([*model.predict(x_test).flat[0:100]])
a = [*y_test[0:100].flat]
b = [*model.predict(x_test).flat[0:100]]
e = np.array(a) - np.array(b)
plot(e,c = "orange")
line_chart.render_to_file('C:\\Users\\1701\Desktop\\Hello_line_chart.svg')


# In[263]:


a = [] #x_train
b = [] #y_train

for i in range(len(y_train)):
    if j>0:
        if i%40  == 0:
            a.append(x_train[i])
            b.append(y_train[i])
plt.figure(figsize=(25,5),dpi=300)
fig = plt.gcf()

plt.plot(b,c='blue',label = "训练集真实值")
plt.plot(model.predict(np.array(a)),c="r",label ="模型输出值")
plt.plot([*(model.predict(np.array(a)) - b).flat],c="lime",label ="温度残差值") 

plt.legend(loc=2, bbox_to_anchor=(0.9,1.0),borderaxespad = 0.,fontsize=25)   
plt.xlabel("特征",fontsize=25)
plt.ylabel("标签",fontsize=25)
  
#plot_as_emf(fig,filename="C:\\Users\\1701\Desktop\\train.emf")


# In[224]:


len(b)


# In[260]:


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


# In[218]:


len(a)


# In[160]:


a = [] #x_train
b = [] #y_train

for i in range(len(y_test)):
    if j>0:
        if i%200  == 0:
            a.append(x_test[i])
            b.append(y_test[i])
plt.figure(figsize=(20,5),dpi=300)
plt.plot(model.predict(np.array(a)),c="blue")
plt.plot(b,c='r')
plt.plot([*(model.predict(np.array(a)) - b).flat],c="black")   


# In[74]:


#! /usr/bin/python
# -*- encoding:utf8 -*-

import numpy as np


def rand(a, b):
    return (b - a) * np.random.random() + a

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BP:
    def __init__(self, layer, iter, max_error):
        self.input_n = layer[0]  # 输入层的节点个数 d
        self.hidden_n = layer[1]  # 隐藏层的节点个数 q
        self.output_n = layer[2]  # 输出层的节点个数 l
        self.gj = []
        self.eh = []
        self.input_weights = []   # 输入层与隐藏层的权值矩阵
        self.output_weights = []  # 隐藏层与输出层的权值矩阵
        self.iter = iter          # 最大迭代次数
        self.max_error = max_error  # 停止的误差范围

        # for i in range(self.input_n + 1):
        #     tmp = []
        #     for j in range(self.hidden_n):
        #         tmp.append(rand(-0.2, 0.2))
        #     self.input_weights.append(tmp)
        #
        # for i in range(self.hidden_n + 1):
        #     tmp = []
        #     for j in range(self.output_n):
        #         tmp.append(rand(-0.2, 0.2))
        #     self.output_weights.append(tmp)
        # self.input_weights = np.array(self.input_weights)
        # self.output_weights = np.array(self.output_weights)

        # 初始化一个(d+1) * q的矩阵，多加的1是将隐藏层的阀值加入到矩阵运算中
        self.input_weights = np.random.random((self.input_n + 1, self.hidden_n))
        # 初始话一个(q+1) * l的矩阵，多加的1是将输出层的阀值加入到矩阵中简化计算
        self.output_weights = np.random.random((self.hidden_n + 1, self.output_n))

        self.gj = np.zeros(layer[2])
        self.eh = np.zeros(layer[1])

    #  正向传播与反向传播
    def forword_backword(self, xj, y, learning_rate=0.1):
        xj = np.array(xj)
        y = np.array(y)
        input = np.ones((1, xj.shape[0] + 1))
        input[:, :-1] = xj
        x = input
        # ah = np.dot(x, self.input_weights)
        ah = x.dot(self.input_weights)
        bh = sigmoid(ah)

        input = np.ones((1, self.hidden_n + 1))
        input[:, :-1] = bh
        bh = input

        bj = np.dot(bh, self.output_weights)
        yj = sigmoid(bj)

        error = yj - y
        self.gj = error * sigmoid_derivative(yj)

        # wg = np.dot(self.output_weights, self.gj)

        wg = np.dot(self.gj, self.output_weights.T)
        wg1 = 0.0
        for i in range(len(wg[0]) - 1):
            wg1 += wg[0][i]
        self.eh = bh * (1 - bh) * wg1
        self.eh = self.eh[:, :-1]

        #  更新输出层权值w，因为权值矩阵的最后一行表示的是阀值多以循环只到倒数第二行
        for i in range(self.output_weights.shape[0] - 1):
            for j in range(self.output_weights.shape[1]):
                self.output_weights[i][j] -= learning_rate * self.gj[0][j] * bh[0][i]

        #  更新输出层阀值b，权值矩阵的最后一行表示的是阀值
        for j in range(self.output_weights.shape[1]):
            self.output_weights[-1][j] -= learning_rate * self.gj[0][j]

        #  更新输入层权值w
        for i in range(self.input_weights.shape[0] - 1):
            for j in range(self.input_weights.shape[1]):
                self.input_weights[i][j] -= learning_rate * self.eh[0][j] * xj[i]

        # 更新输入层阀值b
        for j in range(self.input_weights.shape[1]):
            self.input_weights[-1][j] -= learning_rate * self.eh[0][j]
        return error

    def fit(self, X, y):

        for i in range(self.iter):
            error = 0.0
            for j in range(len(X)):
                error += self.forword_backword(X[j], y[j])
            error = error.sum()
            if abs(error) <= self.max_error:
                break

    def predict(self, x_test):
        x_test = np.array(x_test)
        tmp = np.ones((x_test.shape[0], self.input_n + 1))
        tmp[:, :-1] = x_test
        x_test = tmp
        an = np.dot(x_test, self.input_weights)
        bh = sigmoid(an)
        #  多加的1用来与阀值相乘
        tmp = np.ones((bh.shape[0], bh.shape[1] + 1))
        tmp[:, : -1] = bh
        bh = tmp
        bj = np.dot(bh, self.output_weights)
        yj = sigmoid(bj)
        print (yj)
        return yj

if __name__ == '__main__':
    #  指定神经网络输入层，隐藏层，输出层的元素个数
    layer = [2, 4, 1]
    X = [
            [1, 1],
            [2, 2],
            [1, 2],
            [1, -1],
            [2, 0],
            [2, -1]
        ]
    y = [[0], [0], [0], [1], [1], [1]]

    x_test = [[2, 3],
              [2, 2]]
    # 设置最大的迭代次数，以及最大误差值
    bp = BP(layer, 10000, 0.0001)
    bp.fit(X, y)
    bp.predict(x_test)


# In[62]:


from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense, Lambda
from keras.layers import BatchNormalization, Embedding, Activation, Reshape,Permute,Bidirectional
from keras.layers.merge import Add,Dot
from keras.layers.recurrent import LSTM, GRU
from keras import backend as K
import tensorflow as tf
 
 
 
image_input = Input(shape=(20, 2048), name='image')
recurrent_network = GRU(units=10,return_sequences=True,
                                    name='recurrent_network')(image_input)
 
new_inpu=recurrent_network
splits = Lambda(lambda x: tf.split(x, num_or_size_splits=20, axis=1))(new_inpu)
print(splits[0])
 
list_recur=[]
for i in range(20):
    
    # x = new_inpu[:, i, :]   # AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
    # print(x.shape)
    # x= K.reshape(x,(1,10,)) 
    # x= K.expand_dims(x,axis=-1) 
    # x=Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x=Reshape((1,10))(splits[i])
    print(x)
    # x=Permute((2, 1))(x)
    # print(x)
    recurrent_network = GRU(units=10,return_sequences=True,
                                    name='recurrent_network{}'.format(i))(x)
    # print(recurrent_network)
    list_recur.append(recurrent_network)
 
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(list_recur)
 
# recurrent_network = GRU(units=10,return_sequences=True,
#                                     name='recurrent_network')(image_input)
# print(decoder_outputs)
# output = TimeDistributed(Dense(units=20,
#                                     activation='softmax'),
#                                     name='output')(decoder_outputs)
 
 
model = Model(inputs=image_input, outputs=decoder_outputs)
 
print(model.summary())


# In[46]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[106]:


time0 = time()
model = Sequential()
model.add(layers.Dense(64, input_dim=27, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=["mse"])

md = model.fit(x_train, y_train,
          epochs=27,
          batch_size=100,
         validation_data = (x_test,y_test))

print("模型运行的时间%ds"%(time()-time0))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# In[6]:


a=mean_squared_error(y_test,model.predict(x_test))
b=mean_absolute_error(y_test,model.predict(x_test))
c=r2_score(y_test,model.predict(x_test))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[18]:


res = ([*model.predict(x_test).flat]  - y_test)


# In[35]:


#画训练集残差
plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(20,5),dpi=300)
plot(res[0:1000])
plt.legend(['Test Residual Graph'])
plt.xlabel('len')
plt.ylabel('residual')
plt.show()


# In[31]:


res_ = ([*model.predict(x_train).flat]  - y_train)


# In[34]:


#测试集残差
plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(20,5),dpi=300)
plot(res_[0:1000])
plt.legend(['Trian Residual Graph'])
plt.xlabel('len')
plt.ylabel('residual')
plt.show()


# In[12]:


#获取训练集合测试集的损失历史数值
training_loss  =  md.history['loss']
test_loss = md.history['val_loss']
#为每个epoch创建编号
epoch_count = range(1,len(training_loss) + 1)
plt.figure(dpi=300)
plt.plot(epoch_count,training_loss,'r--')
plt.plot(epoch_count,test_loss,'b--')
plt.legend(['training loss','test loss'])
plt.xlabel('EPOCH')
plt.ylabel('loss')
plt.show()


# In[53]:


pre_y = [*model.predict(x_test).flat] 
plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(20,5),dpi=300)
r = np.array(pre_y) - y_test
#plot(pre_y - y_test.tolist(),c='b')
#plot(y,c = 'b')
plot(abs(r))
plt.legend(['Residual Graph'])
plt.xlabel('len')
plt.ylabel('residual')
plt.show()


# In[68]:


min(abs(r))


# In[4]:


#保存模型
from sklearn.externals import joblib
joblib.dump(model, "ganzhiji_model.m")


# In[5]:


model = joblib.load("ganzhiji_model.m")


# In[97]:


#正常
df = pd.read_csv('C:\\Users\\1701\Desktop\\3808.csv')
df


# In[98]:



X = df.loc[:,['Wspd_min', 'Wspd_max', 'Wspd_avg', 'GenSpd_min', 'GenSpd_max',
       'GenSpd_avg', 'ExlTmp_min', 'ExlTmp_max', 'ExlTmp_avg', 'TurIntTmp_min',
       'TurIntTmp_max', 'TurIntTmp_avg', 'GenAPhsA_min', 'GenAPhsA_max',
       'GenAPhsA_avg', 'TurPwrReact_min', 'TurPwrReact_max', 'TurPwrReact_avg',
       'TurPwrAct_min', 'TurPwrAct_max', 'TurPwrAct_avg', 'WGEN_GnTmpNonDrv_min',
       'WGEN_GnTmpNonDrv_max', 'WGEN_GnTmpNonDrv_avg', 'WROT_PtAngValBl1_min',
       'WROT_PtAngValBl1_max', 'WROT_PtAngValBl1_avg']].values
Y = df.loc[:,'WGEN_GnTmpDrv_avg'].values


# In[99]:


y = model.predict(X) 
r_guzhang = [*y.flat] - Y
len(r)


# In[102]:


abs(r).min()


# In[100]:


a=mean_squared_error(Y,model.predict(X))
b=mean_absolute_error(Y,model.predict(X))
c=r2_score(Y,model.predict(X))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[103]:



plt.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(20,5),dpi=300)
plot(r_guzhang)
plt.legend(['Residual Graph'])
plt.xlabel('len')
plt.ylabel('residual')
plt.show()


# In[104]:


#ADF检验，返回的第一值是ADF指标的值，第二个是P值，接着是时间延迟和样本量。最后一个是词典，给出了样本量的T分布
import numpy as np 
import statsmodels.tsa.stattools as ts
result = ts.adfuller(r, 1)
result


# In[106]:


dff = pd.read_csv('C:\\Users\\1701\Desktop\\3810.csv')
XX = dff.loc[:,['Wspd_min', 'Wspd_max', 'Wspd_avg', 'GenSpd_min', 'GenSpd_max',
       'GenSpd_avg', 'ExlTmp_min', 'ExlTmp_max', 'ExlTmp_avg', 'TurIntTmp_min',
       'TurIntTmp_max', 'TurIntTmp_avg', 'GenAPhsA_min', 'GenAPhsA_max',
       'GenAPhsA_avg', 'TurPwrReact_min', 'TurPwrReact_max', 'TurPwrReact_avg',
       'TurPwrAct_min', 'TurPwrAct_max', 'TurPwrAct_avg', 'WGEN_GnTmpNonDrv_min',
       'WGEN_GnTmpNonDrv_max', 'WGEN_GnTmpNonDrv_avg', 'WROT_PtAngValBl1_min',
       'WROT_PtAngValBl1_max', 'WROT_PtAngValBl1_avg']].values
YY = dff.loc[:,'WGEN_GnTmpDrv_avg'].values
y = model.predict(XX) 
rr = [*y.flat] - YY


# In[109]:


a=mean_squared_error(YY,model.predict(XX))
b=mean_absolute_error(YY,model.predict(XX))
c=r2_score(Y,model.predict(X))
print(str.format('均方误差为:{0:.3f}，平均绝对误差:{1:.3f},R方：{2:.3f}',a,b,c))


# In[113]:


plt.figure(figsize=(20,5))
plot(r_guzhang[0:1000],c='r',label="异常风机残差")
#plot(y,c = 'b')
plot(rr[0:1000],label="正常风机残差")
plt.legend()
plt.xlabel('len')
plt.ylabel('residual')
plt.show()


# In[25]:


a1 = pd.DataFrame(r)
a1.columns = ['cancha']
a2 = df.loc[:,['ProType_avg','timestape_max','time']]
a = pd.concat([a1,a2],axis=1)
a.to_csv('C:\\Users\\1701\Desktop\\3895cancha.csv',index=False)


# In[105]:


def clearn(astr):
    s = astr
    s = ''.join(s.split())
    s = s.replace('=','')
    return s
a= clearn(sfdgse)
a

