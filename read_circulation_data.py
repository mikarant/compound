#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:32:22 2021

@author: rantanem
"""
from pandas.tseries.offsets import MonthEnd
import pandas as pd
import numpy as np
import wget
import os



indices = ['nao','scand','eawr','ea']

out = 'index.txt'

outfile = '/home/rantanem/Documents/python/predict/manuscript/indices.csv'

df_all = []

for i in indices:
    url = 'ftp://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/'+i +'_index.tim'
    
    print(url)
    if os.path.exists(out):
        os.remove(out) # if exist, remove it directly
    index = wget.download(url, out = out)
    
    df = pd.read_csv(out, skiprows=8,  delim_whitespace=True)
    df['Day'] = 1
    df.index = pd.to_datetime(df[['YEAR','MONTH','Day']]) + MonthEnd(1)
    df.drop(columns=['YEAR','MONTH','Day'], inplace=True)
    cond = (df.index.year >=1961) & (df.index.year < 2020)
    df = df[cond]
    
    df.rename(columns={'INDEX':i}, inplace=True)
    
    df_all.append(df)
    
df_all = pd.concat(df_all, axis=1)


## separate treatment for AO index
# filename + path
path_ao = '/home/rantanem/Documents/python/predict/monthly_ao.csv'
ao_array = np.genfromtxt(path_ao,skip_header=0,delimiter="")[:,1:13]
years = np.genfromtxt(path_ao,skip_header=0,delimiter="")[:,0]
df_ao = pd.DataFrame(ao_array.flatten(),columns=['ao'])
df_ao.index = df_all.index


df_all = pd.concat([df_all, df_ao], axis=1)

### output 
df_all.to_csv(outfile)
