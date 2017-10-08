# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:04:24 2017

@author: Abner Bian
"""

import pandas as pd
import numpy as np
hdd=pd.read_csv('C:/Users/admin/Desktop/data.csv')
###将故障样本取出来
hdd_failure=hdd[hdd['failure']==1]
#去除所有故障的序列号
hdd_failure_serial_number=hdd_failure['serial_number']
list_=[]
data_new=pd.DataFrame()
##找出所有故障的序列号的记录
for i in range(0,hdd_failure_serial_number.shape[0]):
    #找到一个故障硬盘的所有记录
    hdd_failure_temp=hdd[hdd['serial_number']==hdd_failure_serial_number.iloc[i]]
    #去除该记录的硬盘样本
    hdd=hdd[hdd['serial_number']!=hdd_failure_serial_number.iloc[i]]
    #设置故障前5天的故障样本
    
    #判断是否有5天，如果>=5,取最后5天的样本，如果<5,全部置一
    count=hdd_failure_temp.shape[0]
    if(count>=5):
        #取最后5个
        for j in range(count-5,count-1):
            hdd_failure_temp.iloc[j,1]=1
    if(count<5):
        #全部置一
        for j in range(0,count-1):
            hdd_failure_temp.iloc[j,1]=1
    list_.append(hdd_failure_temp)
frame = pd.concat(list_)
    
    
    

















