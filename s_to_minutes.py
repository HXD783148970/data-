# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:15:36 2019
将秒级数据转化为十分钟数据，先将时间转化为时间戳，然后按时间戳列进行分段求均值，最大值，最小值，最后将三个列表合并为一个列表,
@author: 1701
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime,time
#data = pd.read_csv('C:/Users/1701/Desktop/2.csv',index_col=False)
data = pd.read_csv('C:/Users/1701/Desktop//WT03871.csv',index_col=False,error_bad_lines=False)
#转化为时间戳
data['temp'] = pd.to_datetime(data.RecTm)
#data['temp'] = pd.to_datetime(data.RecTm,errors = 'ignore')#忽略错误行
data['temp1'] = data.temp.astype('str')
#################################################
def convert(x):
    d = datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp
###################################################
data['timestamp'] = data.temp1.apply(lambda x: convert(x))
#将时间列作为索引
#data.set_index("timestamp", inplace=True)
data.drop(['RecTm','temp','temp1'],axis=1,inplace= True)
#按升序进行排序
data.sort_index(inplace=True)

a=[]#每个子区间
b=[]#平均值
c=[]#最大值
d=[]#最小值
ndate=0
mtime = 600
starttime=0
endtime  =0
gap=0
mean_value=[]
max_value=[]
min_value=[]
for i in range(0,len(data)):
    ntime = data.loc[i,"timestamp"]
    if starttime == 0:
        print("-----------开始------------")
        starttime = ntime
        endtime   = ntime + mtime
        
    if starttime < ntime and endtime >= ntime:
        a.append(data.loc[i,:])
        #("aaaaaaaaaaaa")
        
    if ntime > endtime:
        gap += 1
        print("-----------开始分组------------"+str(gap) + "---len:"+str(len(a)))
        starttime += mtime
        endtime   += mtime
        #a=pd.DataFrame(a)
        if a:
        # 存在值即为真
            x=np.mean(a[:],axis=0)
            mean_value.append(x)

            y=np.max(a[:],axis=0)
            max_value.append(y)
        
            z=np.min(a[:],axis=0)
            min_value.append(z)
        else:
            print(a)
        a.clear()

print("----------处理完成----------")

colums_max = ['ProType_max', 'TurItvWithApt_max', 'TurStatus_max', 'Wspd_max', 'GenSpd_max',
       'TurRotSpd_max', 'Wdir_max', 'TurYawDir_max', 'GbxOilTmp_max', 'GbxShfTmp_max',
       'ExlTmp_max', 'TurIntTmp_max', 'GenGnTmp_max', 'GenAPhsA_max', 'GenAPhsB_max',
       'GenAPhsC_max', 'GenVPhsA_max', 'GenVPhsB_max', 'GenVPhsC_max', 'GenHz_max',
       'TurPwrReact_max', 'TurPwrAct_max', 'TurPF_max', 'GenTotPwr_max', 'GenTotTm_max',
       'TurFltTm_max', 'TurStdbyTm_max', 'WCNV_CnvTmp_max', 'WCNV_Torq_max',
       'WCNV_TorqSp_max', 'WGEN_GnTmpDrv_max', 'WGEN_GnTmpNonDrv_max', 'WNAC_Dir_max',
       'WNAC_DispXdirL_max', 'WNAC_DispYdirV_max', 'WNAC_IntlTmpCct_max',
       'WNAC_WdDir_max', 'WROT_HubTmp_max', 'WROT_PtAngValBl1_max',
       'WROT_PtAngValBl2_max', 'WROT_PtAngValBl3_max', 'WROT_PtGmTmp1_max',
       'WROT_PtGmTmp2_max', 'WROT_PtGmTmp3_max', 'WROT_PtTmpCct1_max',
       'WROT_PtTmpCct2_max', 'WROT_PtTmpCct3_max', 'WTRM_BrkHyPres_max',
       'WTRM_GbxOilPres_max', 'WTRM_TrmTmpEntGbxNonDrv_max', 'WTRM_TrmTmpShfBrg_max',
       'WTUR_TotVArh_max', 'WTUR_TotVArhCons_max', 'WTUR_TotWhCons_max',
       'WTUR_TurFltCd_max', 'CONT_TEMP_max', 'WYAW_CcWSt_max', 'WYAW_CWSt_max','timestape_max']

colums_min = ['ProType_min', 'TurItvWithApt_min', 'TurStatus_min', 'Wspd_min', 'GenSpd_min',
       'TurRotSpd_min', 'Wdir_min', 'TurYawDir_min', 'GbxOilTmp_min', 'GbxShfTmp_min',
       'ExlTmp_min', 'TurIntTmp_min', 'GenGnTmp_min', 'GenAPhsA_min', 'GenAPhsB_min',
       'GenAPhsC_min', 'GenVPhsA_min', 'GenVPhsB_min', 'GenVPhsC_min', 'GenHz_min',
       'TurPwrReact_min', 'TurPwrAct_min', 'TurPF_min', 'GenTotPwr_min', 'GenTotTm_min',
       'TurFltTm_min', 'TurStdbyTm_min', 'WCNV_CnvTmp_min', 'WCNV_Torq_min',
       'WCNV_TorqSp_min', 'WGEN_GnTmpDrv_min', 'WGEN_GnTmpNonDrv_min', 'WNAC_Dir_min',
       'WNAC_DispXdirL_min', 'WNAC_DispYdirV_min', 'WNAC_IntlTmpCct_min',
       'WNAC_WdDir_min', 'WROT_HubTmp_min', 'WROT_PtAngValBl1_min',
       'WROT_PtAngValBl2_min', 'WROT_PtAngValBl3_min', 'WROT_PtGmTmp1_min',
       'WROT_PtGmTmp2_min', 'WROT_PtGmTmp3_min', 'WROT_PtTmpCct1_min',
       'WROT_PtTmpCct2_min', 'WROT_PtTmpCct3_min', 'WTRM_BrkHyPres_min',
       'WTRM_GbxOilPres_min', 'WTRM_TrmTmpEntGbxNonDrv_min', 'WTRM_TrmTmpShfBrg_min',
       'WTUR_TotVArh_min', 'WTUR_TotVArhCons_min', 'WTUR_TotWhCons_min',
       'WTUR_TurFltCd_min', 'CONT_TEMP_min', 'WYAW_CcWSt_min', 'WYAW_CWSt_min','timestape_min']
colums_avg=['ProType_avg', 'TurItvWithApt_avg', 'TurStatus_avg', 'Wspd_avg', 'GenSpd_avg',
       'TurRotSpd_avg', 'Wdir_avg', 'TurYawDir_avg', 'GbxOilTmp_avg', 'GbxShfTmp_avg', 'ExlTmp_avg',
       'TurIntTmp_avg', 'GenGnTmp_avg', 'GenAPhsA_avg', 'GenAPhsB_avg', 'GenAPhsC_avg', 'GenVPhsA_avg',
       'GenVPhsB_avg', 'GenVPhsC_avg', 'GenHz_avg', 'TurPwrReact_avg', 'TurPwrAct_avg', 'TurPF_avg',
       'GenTotPwr_avg', 'GenTotTm_avg', 'TurFltTm_avg', 'TurStdbyTm_avg', 'WCNV_CnvTmp_avg',
       'WCNV_Torq_avg', 'WCNV_TorqSp_avg', 'WGEN_GnTmpDrv_avg', 'WGEN_GnTmpNonDrv_avg',
       'WNAC_Dir_avg', 'WNAC_DispXdirL_avg', 'WNAC_DispYdirV_avg', 'WNAC_IntlTmpCct_avg',
       'WNAC_WdDir_avg', 'WROT_HubTmp_avg', 'WROT_PtAngValBl1_avg', 'WROT_PtAngValBl2_avg',
       'WROT_PtAngValBl3_avg', 'WROT_PtGmTmp1_avg', 'WROT_PtGmTmp2_avg', 'WROT_PtGmTmp3_avg',
       'WROT_PtTmpCct1_avg', 'WROT_PtTmpCct2_avg', 'WROT_PtTmpCct3_avg', 'WTRM_BrkHyPres_avg',
       'WTRM_GbxOilPres_avg', 'WTRM_TrmTmpEntGbxNonDrv_avg', 'WTRM_TrmTmpShfBrg_avg',
       'WTUR_TotVArh_avg', 'WTUR_TotVArhCons_avg', 'WTUR_TotWhCons_avg', 'WTUR_TurFltCd_avg',
       'CONT_TEMP_avg', 'WYAW_CcWSt_avg', 'WYAW_CWSt_avg','timestape_avg']
#mean_value = [i for i in mean_value if i != []]#删除列表中的空子列表
#max_value = [i for i in max_value if i != []]
#min_value = [i for i in min_value if i != []]
A=pd.DataFrame(mean_value)
B=pd.DataFrame(max_value)   
C=pd.DataFrame(min_value)
A.columns = colums_avg   
B.columns = colums_max
C.columns = colums_min   


list=[A,B,C]
df=pd.concat(list,axis=1)

#df.to_csv('C:/Users/1701/Desktop/WT10_min.csv',index=False)
new_df=[[] for q in range(34)] 
for line in range(len(df)): 
    
    if (
        df.ix[line,'Wspd_min'] >=3.5 and
        df.ix[line,'Wspd_min'] <=50 and
        df.ix[line,'Wspd_max'] >=3.5 and
        df.ix[line,'Wspd_max'] <=50 and
        df.ix[line,'Wspd_avg'] >=3.5 and
        df.ix[line,'Wspd_avg'] <=50 and
        df.ix[line,'GenSpd_min'] >=0 and
        df.ix[line,'GenSpd_min'] <=2000 and
        df.ix[line,'GenSpd_max'] >=0 and
        df.ix[line,'GenSpd_max'] <=2000 and
        df.ix[line,'GenSpd_avg'] >=0 and
        df.ix[line,'GenSpd_avg'] <=2000 and
        df.ix[line,'ExlTmp_min'] >=-40 and
        df.ix[line,'ExlTmp_min'] <=50 and
        df.ix[line,'ExlTmp_max'] >=-40 and
        df.ix[line,'ExlTmp_max'] <=50 and
        df.ix[line,'ExlTmp_avg'] >=-40 and
        df.ix[line,'ExlTmp_avg'] <=50 and
        df.ix[line,'TurIntTmp_min'] >=-40 and
        df.ix[line,'TurIntTmp_min'] <=60 and
        df.ix[line,'TurIntTmp_max'] >=-40 and
        df.ix[line,'TurIntTmp_max'] <=60 and
        df.ix[line,'TurIntTmp_avg'] >=-40 and
        df.ix[line,'TurIntTmp_avg'] <=60 and
        df.ix[line,'GenAPhsA_min'] >=0 and
        df.ix[line,'GenAPhsA_min'] <=1500 and
        df.ix[line,'GenAPhsA_max'] >=0 and
        df.ix[line,'GenAPhsA_max'] <=1500 and
        df.ix[line,'GenAPhsA_avg'] >=0 and
        df.ix[line,'GenAPhsA_avg'] <=1500 and
        df.ix[line,'TurPwrReact_min'] >=-200 and
        df.ix[line,'TurPwrReact_min'] <=500 and
        df.ix[line,'TurPwrReact_max'] >=-200 and
        df.ix[line,'TurPwrReact_max'] <=500 and
        df.ix[line,'TurPwrReact_avg'] >=-200 and
        df.ix[line,'TurPwrReact_avg'] <=500 and
        df.ix[line,'TurPwrAct_min'] >0 and
        df.ix[line,'TurPwrAct_min'] <=2000 and
        df.ix[line,'TurPwrAct_max'] >0 and
        df.ix[line,'TurPwrAct_max'] <=2000 and
        df.ix[line,'TurPwrAct_avg'] >0 and
        df.ix[line,'TurPwrAct_avg'] <=2000 and
        df.ix[line,'WGEN_GnTmpDrv_min'] >=-10 and
        df.ix[line,'WGEN_GnTmpDrv_min'] <=90 and
        df.ix[line,'WGEN_GnTmpDrv_max'] >=-10 and
        df.ix[line,'WGEN_GnTmpDrv_max'] <=90 and
        df.ix[line,'WGEN_GnTmpDrv_avg'] >=-10 and
        df.ix[line,'WGEN_GnTmpDrv_avg'] <=90 and
        df.ix[line,'WGEN_GnTmpNonDrv_min'] >=-10 and
        df.ix[line,'WGEN_GnTmpNonDrv_min'] <=90 and
        df.ix[line,'WGEN_GnTmpNonDrv_max'] >=-10 and
        df.ix[line,'WGEN_GnTmpNonDrv_max'] <=90 and
        df.ix[line,'WGEN_GnTmpNonDrv_avg'] >=-10 and
        df.ix[line,'WGEN_GnTmpNonDrv_avg'] <=90 and
        df.ix[line,'WROT_PtAngValBl1_min'] >=-5 and
        df.ix[line,'WROT_PtAngValBl1_min'] <=90 and
        df.ix[line,'WROT_PtAngValBl1_max'] >=-5 and
        df.ix[line,'WROT_PtAngValBl1_max'] <=90 and
        df.ix[line,'WROT_PtAngValBl1_avg'] >=-5 and
        df.ix[line,'WROT_PtAngValBl1_avg'] <=90 
    
       ):

        new_df[0].append(df.ix[line,'ProType_avg'])
        new_df[1].append(df.ix[line,'Wspd_min'])
        new_df[2].append(df.ix[line,'Wspd_max'])
        new_df[3].append(df.ix[line,'Wspd_avg'])
        new_df[4].append(df.ix[line,'GenSpd_min'])
        new_df[5].append(df.ix[line,'GenSpd_max'])
        new_df[6].append(df.ix[line,'GenSpd_avg'])
        new_df[7].append(df.ix[line,'ExlTmp_min'])
        new_df[8].append(df.ix[line,'ExlTmp_max'])
        new_df[9].append(df.ix[line,'ExlTmp_avg'])
        new_df[10].append(df.ix[line,'TurIntTmp_min'])
        new_df[11].append(df.ix[line,'TurIntTmp_max'])
        new_df[12].append(df.ix[line,'TurIntTmp_avg'])
        new_df[13].append(df.ix[line,'GenAPhsA_min'])
        new_df[14].append(df.ix[line,'GenAPhsA_max'])
        new_df[15].append(df.ix[line,'GenAPhsA_avg'])
        new_df[16].append(df.ix[line,'TurPwrReact_min'])
        new_df[17].append(df.ix[line,'TurPwrReact_max'])
        new_df[18].append(df.ix[line,'TurPwrReact_avg'])
        new_df[19].append(df.ix[line,'TurPwrAct_min'])
        new_df[20].append(df.ix[line,'TurPwrAct_max'])
        new_df[21].append(df.ix[line,'TurPwrAct_avg'])
        new_df[22].append(df.ix[line,'WGEN_GnTmpDrv_min'])
        new_df[23].append(df.ix[line,'WGEN_GnTmpDrv_max'])
        new_df[24].append(df.ix[line,'WGEN_GnTmpDrv_avg'])
        new_df[25].append(df.ix[line,'WGEN_GnTmpNonDrv_min'])
        new_df[26].append(df.ix[line,'WGEN_GnTmpNonDrv_max'])
        new_df[27].append(df.ix[line,'WGEN_GnTmpNonDrv_avg']) 
        new_df[28].append(df.ix[line,'WROT_PtAngValBl1_min'])
        new_df[29].append(df.ix[line,'WROT_PtAngValBl1_max'])
        new_df[30].append(df.ix[line,'WROT_PtAngValBl1_avg'])
        new_df[31].append(df.ix[line,'timestape_min'])
        new_df[32].append(df.ix[line,'timestape_max'])
        new_df[33].append(df.ix[line,'timestape_avg'])


w=pd.DataFrame(new_df).T
#w.ix[:,19].values
plt.scatter(w.ix[:,3].values,w.ix[:,21].values)

#填充空值为0
#填充所有的空值为0
w=w.fillna(0)

colums_label=[
 'ProType_avg'
,'Wspd_min'
,'Wspd_max'
,'Wspd_avg'
,'GenSpd_min'
,'GenSpd_max'
,'GenSpd_avg'
,'ExlTmp_min'
,'ExlTmp_max'
,'ExlTmp_avg'
,'TurIntTmp_min'
,'TurIntTmp_max'
,'TurIntTmp_avg'
,'GenAPhsA_min'
,'GenAPhsA_max'
,'GenAPhsA_avg'
,'TurPwrReact_min'
,'TurPwrReact_max'
,'TurPwrReact_avg'
,'TurPwrAct_min'
,'TurPwrAct_max'
,'TurPwrAct_avg'
,'WGEN_GnTmpDrv_min'
,'WGEN_GnTmpDrv_max'
,'WGEN_GnTmpDrv_avg'
,'WGEN_GnTmpNonDrv_min'
,'WGEN_GnTmpNonDrv_max'
,'WGEN_GnTmpNonDrv_avg'
,'WROT_PtAngValBl1_min'
,'WROT_PtAngValBl1_max'
,'WROT_PtAngValBl1_avg'
,'timestape_min'
,'timestape_max'
,'timestape_avg'
]
w.columns = colums_label

#将时间戳数据转为时间数据
def timeStamp(timeNum):
    timeStamp = float(timeNum)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
    print (timeStamp,otherStyleTime)
    return otherStyleTime

w["time"]= w.timestape_max.apply(lambda x: timeStamp(x))   
#保存数据
w.to_csv('C:/Users/1701/Desktop/WT10_min过滤数据.csv',index=False)
      
     
