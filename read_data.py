#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:47:35 2020

@author: rantanem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import xarray as xr
import functions as fc
from pyproj import Proj
import sys
pd.options.mode.chained_assignment = None  # default='warn'

# if command-line arguments are not given, use default data
if len( sys.argv ) != 2:
    place = 'kemi'
    print(sys.argv[0],'Give place as a command-line argument. Now using ' +place.capitalize())
    
else:
    place = sys.argv[1]
    

## datapoths
path = '/home/rantanem/Documents/python/predict/dates/fintidegauges/'
output_path = '/home/rantanem/Documents/python/predict/dates/'

## original data consists data up to 2018. Additional data to cover 2019
filename = place + 'sl.txt'
filename_2019 = place + '_tunti2019.txt'
filepath = path + filename

## datafile for precipitation
gridfile ="/home/rantanem/Downloads/daily_avg_rel_hum_1961_2018_netcdf/rrday.nc"

# define tide gauge coordinates
mareograph_coordinates = {'kemi' :        [65.67, 24.52],
                          'oulu' :        [65.04, 25.42],
                          'kaskinen' :    [62.34, 21.21],
                          'pietarsaari' : [63.71, 22.69],
                          'rauma' :       [61.13, 21.43],
                          'föglö':        [60.03, 20.38],
                          'hanko':        [59.82, 22.98],
                          'helsinki':     [60.15, 24.96],
                          'hamina':       [60.56, 27.18]
                          }

if place in mareograph_coordinates:
    print(place.capitalize() + ' is valid place. Continuing...')
else:
    print('Place is not valid. Give one of these:')
    print(list(mareograph_coordinates.keys()))
    sys.exit()

# define the study period
startYear = 1961
endYear = 2019

# percentiles for the threshold values:
per_elevated = 0.95
per_high = 0.99




### read the sea level data
data = fc.read_sl_data(filepath, startYear)

additional_data = pd.DataFrame(np.genfromtxt(path + filename_2019))
additional_data = fc.modify_data(additional_data,'Sea_level')

# concatenate data
allData = pd.concat([data, additional_data])


# make daily means and maximums using 6 UTC-6UTC period
allData.index  = allData.index + pd.to_timedelta('-6H')
dailymeans = allData.groupby(pd.Grouper(freq='1D')).mean()
dailymax = allData.groupby(pd.Grouper(freq='1D')).max()

# calculate annual means
yearlymeans = dailymeans.groupby(pd.Grouper(freq='1Y')).mean()



# fig = plt.figure(figsize=(10,5),dpi=120)

# plt.plot(yearlymeans)
# plt.xlim(pd.to_datetime('1960-1-1'), pd.to_datetime('2020-12-31'))

# calculate the number of missing days
missing_days = np.sum(np.isnan(dailymax))
missing_days_in_percents = np.round((np.sum(np.isnan(dailymax)) / len(dailymax))*100,1)

print('Missing days: ' + str( missing_days.values[0]))
print('Missing days in percents: ' + str( missing_days_in_percents.values[0]))


missing_data_by_year = np.isnan(dailymax).groupby(dailymax.index.year).sum()
# fig = plt.figure(figsize=(10,5),dpi=120)
# plt.bar(missing_data_by_year.index, missing_data_by_year.values.squeeze())
# plt.title('Missing data by year')
# plt.ylabel('Number of missing days')

missing_data_by_month = np.isnan(dailymax).groupby(dailymax.index.month).sum()
# fig = plt.figure(figsize=(10,5),dpi=120)
# plt.bar(missing_data_by_month.index, missing_data_by_month.values.squeeze())
# plt.title('Missing data by month')
# plt.ylabel('Number of missing days')



# ### calculate linear trend in daily means
y = np.array(dailymeans.Sea_level.values)
x = mdates.date2num(dailymeans.index.values)
idx = ~np.isnan(y)

z = np.polyfit(x[idx], y[idx], 1)
p = np.poly1d(z)
xx = np.linspace(x.min(), x.max(), len(dailymax))
print('Trend in dailymeans: ' + str(np.round(p.c[0]*20*365/10,2)) + ' cm/10y')



### calculate linear trend in yearly means
y = np.array(yearlymeans.Sea_level.values)
x = mdates.date2num(yearlymeans.index.values)
idx = ~np.isnan(y)

z = np.polyfit(x[idx], y[idx], 1)
p2 = np.poly1d(z)


# print('Trend in yearly means: ' + str(np.round(p2.c[0]*20*365/10,2)) + ' cm/10y')


### remove the *yearly* trend from daily maximum values
dailymax.Sea_level = dailymax.Sea_level - p2(xx)
dailymeans.Sea_level = dailymeans.Sea_level - p2(xx)


################################
## precipitation

dset = xr.open_dataset(gridfile)


finproj = Proj("epsg:3067")

x, y = finproj(mareograph_coordinates[place][1], mareograph_coordinates[place][0], inverse=False)


### select mean precipitation averaged over the grid points 
### +-30 km around the mareograph
dist = 30000
inc = 10000
new_x = np.arange(x-dist, x+dist+inc, inc)
new_y = np.arange(y-dist, y+dist+inc, inc)
prec1 = dset.interp(Lat=new_y, Lon=new_x, method = 'nearest').RRday.mean(dim='Lat').mean(dim='Lon')
prec1 = prec1.to_dataframe()


### concatenate with daily sea levels and drop NaN's
prec1.index = prec1.index + pd.to_timedelta('0D') 
a = pd.concat([dailymax,prec1.RRday], join='outer', axis=1).dropna()



### rename column
a.rename(columns={'RRday':'Precipitation'}, inplace=True)
### round sea levels to centimeters
a['Sea_level'] = a['Sea_level'].round()/10
### round precipitation to millimeters
a['Precipitation'] = a['Precipitation'].round(1)
### add label to dates
a.index.name = 'Date'




preclevel0 = np.round(a.Precipitation.quantile(0.98),2)
sealevel0 = np.round(a.Sea_level.quantile(0.98),0)
preclevel1 = np.round(a.Precipitation.quantile(per_elevated),2)
sealevel1 = np.round(a.Sea_level.quantile(per_elevated),0)
preclevel2 = np.round(a.Precipitation.quantile(per_high),2)
sealevel2 = np.round(a.Sea_level.quantile(per_high),0)


print('\nPrecipitation 90th: ' + str(preclevel1 ))
print('Precipitation 98th: ' + str(preclevel2 ))
print('Sea level 90th: ' + str(sealevel1 ))
print('Sea level 98th: ' + str(sealevel2 ))
print(' ')

output_path = output_path  + place + '/'



#### Condition 1: compound elevated level
comp_elevated = (a.Sea_level >= sealevel1) & (a.Precipitation >= preclevel1)

 
## take only independent events (i.e. remove consecutive events)
df_elev = fc.independent_events(a[comp_elevated], 'Sea_level')
df_elev.to_csv(output_path + 'Dates_compound_elevated_' + place + '_grid.csv')



#### Condition 2: compound high level
comp_high = (a.Sea_level >= sealevel2) & (a.Precipitation >= preclevel2) 
df_high = fc.independent_events(a[comp_high], 'Sea_level')
df_high.to_csv(output_path + 'Dates_compound_high_' + place + '_grid.csv')




### non-compound events

##### high & elevated sea level alone
high_sealevel = (a.Sea_level >= sealevel2) & (a.Precipitation < preclevel2)
df_high_sl = fc.independent_events(a[high_sealevel], 'Sea_level')

df_high_sl.to_csv(output_path + 'Dates_sea_level_high_' + place + '_grid.csv')

elevated_sealevel = (a.Sea_level >= sealevel1) & (a.Precipitation < preclevel0)
df_elev_sl = fc.independent_events(a[elevated_sealevel], 'Sea_level')

df_elev_sl.to_csv(output_path + 'Dates_sea_level_elevated_' + place + '_grid.csv')

##### high & elevated precipitation alone
high_prec = (a.Precipitation >= preclevel0) & (a.Sea_level < sealevel2)
df_high_prec = fc.independent_events(a[high_prec], 'Precipitation')
df_high_prec.to_csv(output_path + 'Dates_prec_high_' + place + '_grid.csv')

elevated_prec = (a.Precipitation >= preclevel1)  & (a.Sea_level < sealevel0)
df_elev_prec = fc.independent_events(a[elevated_prec], 'Precipitation')
df_elev_prec.to_csv(output_path + 'Dates_prec_elevated_' + place + '_grid.csv')



##### high & elevated sea level
# high_sealevel = (a.Sea_level >= sealevel2) 

# a[high_sealevel].to_csv(output_path + 'Dates_sea_level_all_high_' + place + '_grid.csv')

# elevated_sealevel = (a.Sea_level >= sealevel1)

# a[elevated_sealevel].to_csv(output_path + 'Dates_sea_level_all_elevated_' + place + '_grid.csv')


# ##### high & elevated precipitation 
# high_prec = (a.Precipitation >= preclevel2) 

# a[high_prec].to_csv(output_path + 'Dates_prec_all_high_' + place + '_grid.csv')

# elevated_prec = (a.Precipitation >= preclevel1)

# a[elevated_prec].to_csv(output_path + 'Dates_prec_all_elevated_' + place + '_grid.csv')

### monthly basis
g = comp_elevated.groupby(pd.Grouper(freq="M")).sum()
g.to_csv(output_path + 'Months_compound_elevated_' + place + '_grid.csv')

g = elevated_sealevel.groupby(pd.Grouper(freq="M")).sum()
g.to_csv(output_path + 'Months_sea_level_elevated_' + place + '_grid.csv')

g = elevated_prec.groupby(pd.Grouper(freq="M")).sum()
g.to_csv(output_path + 'Months_prec_elevated_' + place + '_grid.csv')


### all precipitation and sea-level

a.to_csv(output_path + 'Dates_all_' + place + '_grid.csv')


