# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:33:10 2017

@author: Abner Bian
"""
import numpy as np
import pandas as pd
import glob
path =r'C:\Users\admin\Desktop\data_Q1_2017' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)