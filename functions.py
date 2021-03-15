#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:30:50 2021

@author: rantanem
"""
import pandas as pd
import numpy as np

def read_sl_data(filepath, startYear):
    
    data = pd.DataFrame(np.genfromtxt(filepath))
    data.columns = ["Year", "Month", "Day", "Hour","Sea_level"]

    data = data[(data.Year>=startYear)]

    data.index =  pd.to_datetime(data[['Year','Month','Day','Hour']])

    data = data.drop(columns=['Year','Month','Day','Hour'])
    
    return data

def modify_data(df, valuename='Value'):
    
    df.columns = ["Year", "Month", "Day", "Hour", valuename]
    df.index =  pd.to_datetime(df[['Year','Month','Day','Hour']])
    df = df.drop(columns=['Year','Month','Day','Hour'])
    
    return(df)