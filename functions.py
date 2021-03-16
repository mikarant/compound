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

def independent_events(df, maxvar):
    
    
    diff = df.index.to_series().diff()
    df["stormID"] = np.nan
    df["Date"] = df.index


    stormID = 0
    for d in df.index:
        D = diff[d]

        if (D > pd.Timedelta(1, unit='D')) | (pd.isnull(D)):
            stormID +=  1
            df['stormID'][d] = stormID
        else:
            df['stormID'][d] = stormID

    new_df = df.groupby(df['stormID']).max(maxvar)
    new_df.index = df['Date'].groupby(df['stormID']).max('Sea_level')
    
    return new_df
