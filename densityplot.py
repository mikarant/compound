#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:19:54 2020

@author: rantanem
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10
def rounddown(x):
    return int(math.floor(x / 10.0)) * 10

fontsize = 18

path = '/home/rantanem/Documents/python/predict/dates/'

place = 'kemi'
filename = 'Dates_'



filepath = path + place + '/' + filename + 'all_' + place + '_grid.csv'
df = pd.read_csv(filepath)

xmax = roundup(df.Precipitation.max())+10

# years
N = len(np.unique(pd.DatetimeIndex(df.Date).year))

# percentiles for the threshold values:
per_elevated = 0.95
per_high = 0.99

# for non-compound events
per_noncompound = 0.5

# levels defined using percentiles
prec_level0 = np.round(df.Precipitation.quantile(0.99),2)
sea_level0 = np.round(df.Sea_level.quantile(0.99),0)
sea_level00 = np.round(df.Sea_level.quantile(0.5),0)
prec_level1 = np.round(df.Precipitation.quantile(per_elevated),2)
sea_level1 = np.round(df.Sea_level.quantile(per_elevated),0)
prec_level2 = np.round(df.Precipitation.quantile(per_high),2)
sea_level2 = np.round(df.Sea_level.quantile(per_high),0)

# p1 = np.round(df.Sea_level.quantile(0.90),2)
# p2 = np.round(df.Sea_level.quantile(0.95),2)
# p3 = np.round(df.Sea_level.quantile(0.99),2)
# print('90 %: ' + str(np.round(p1,0)))
# print('95 %: ' + str(np.round(p2,0)))
# print('99 %: ' + str(np.round(p3,0)))



preclevel1 = prec_level1 
preclevel2 = prec_level2 
sealevel1 = sea_level1 
sealevel2 = sea_level2 




# bin interval
int_prec = 2
int_sea = 5

# bins for plotting
bins_prec = np.arange(0,xmax,int_prec)
bins_sea = np.arange(-120,200,int_sea)


#### PLOT ####

fig= plt.figure(figsize=(7,5),dpi=120)

h = plt.hist2d(df.Precipitation, df.Sea_level, bins=[bins_prec, bins_sea], 
               norm=mpl.colors.LogNorm(), cmap='gist_ncar', 
               weights=(1/N)*np.ones_like(df.Sea_level.values))
ax=plt.gca()
cb = plt.colorbar(h[3], ax=ax)
cb.set_label(label='Events per year',fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)

plt.xlim(0,roundup(df.Precipitation.max())+5)
plt.ylim(rounddown(df.Sea_level.min())-10,roundup(df.Sea_level.max())+10)
plt.grid()

plt.ylabel('Sea level anomaly [cm]', fontsize=fontsize)
plt.xlabel('Precipitation [mm]', fontsize=fontsize)



ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.tick_params(axis='both', which='minor', labelsize=fontsize)

# find the number of cases in each category

# bins for histogram
bins_prec_hist = np.arange(0,90,0.1)
bins_sea_hist = np.arange(-120,210,0.1)

H, xedges, yedges = np.histogram2d(df.Precipitation, df.Sea_level,bins=[bins_prec_hist, bins_sea_hist])
                      
x1 = np.where(np.isclose(bins_prec_hist, preclevel1, atol=1e-6))[0][0]
x2 = len(bins_prec_hist) 
y1 = np.where(np.isclose(bins_sea_hist, sealevel1, atol=1e-6))[0][0]
y2 = len(bins_sea_hist) 

n1 = np.round(np.sum(H[x1:x2, y1:y2])/N, 2)

x1 = np.where(np.isclose(bins_prec_hist, preclevel2, atol=0.1))[0][0]
x2 = len(bins_prec_hist) 
y1 = np.where(np.isclose(bins_sea_hist, sealevel2, atol=1e-6))[0][0]
y2 = len(bins_sea_hist) 

n2 = np.round(np.sum(H[x1:x2, y1:y2])/N, 2)

x1 = np.where(np.isclose(bins_prec_hist, prec_level0, atol=0.1))[0][0]
x2 = len(bins_prec_hist) 
y1 = np.where(np.isclose(bins_sea_hist, sea_level0, atol=1e-6))[0][0]
y2 = 0

n3 = np.round(np.sum(H[x1:x2, y2:y1])/N, 2)

x1 = np.where(np.isclose(bins_prec_hist, prec_level0, atol=0.1))[0][0]
x2 = 0
y1 = np.where(np.isclose(bins_sea_hist, sea_level0, atol=1e-6))[0][0]
y2 = len(bins_sea_hist) 

n4 = np.round(np.sum(H[x2:x1, y1:y2])/N, 2)



# plt.plot(np.linspace(preclevel1,xmax,50), (sealevel1)*np.ones(50), 'k--', label='Elevated: ' + str(n1) + ' events year⁻¹')
# plt.plot(preclevel1*np.ones(30), np.linspace(sealevel1,210,30), 'k--')

plt.plot(np.linspace(preclevel2,xmax,100), sea_level00*np.ones(100), 'k--', label='High: ' + str(n2) + ' events year⁻¹')
plt.plot(preclevel2*np.ones(100), np.linspace(sealevel2,210,100), 'k--')

# plt.plot(np.linspace(prec_level0,80,100), sea_level0*np.ones(100), 'r--', label='Non-compound: ' + str(n3) + ' events year⁻¹')
# plt.plot(prec_level0*np.ones(100), np.linspace(210,-100,100), 'r-')

# plt.plot(np.linspace(0,prec_level0,100), sea_level0*np.ones(100), 'r--', label='Non-compound: ' + str(n3) + ' events year⁻¹')
# plt.plot(prec_level0*np.ones(100), np.linspace(sea_level0,-100,100), 'r--')

## vertical high line
plt.plot(prec_level0*np.ones(100), np.linspace(210,-110,100), 'r-')

## horizontal high line
plt.plot(np.linspace(0,80,100), sea_level0*np.ones(100), 'r-')

plt.annotate('Zone 1', (30,150), color='r')
# plt.annotate('Zone 2', (50,70), color='k')
plt.annotate('Zone 2', (30,-40), color='r')
plt.annotate('Zone 3', (1,165), color='r')

plt.title(place.capitalize(), fontsize=fontsize)
# plt.legend(loc='upper right', fontsize=10)

figurePath = '/home/rantanem/Documents/python/predict/figures/'
figureName = 'densityplot_' + place + '.png'
   
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


R = np.corrcoef(df.Precipitation, df.Sea_level)
print('Correlation: ' + str(np.round(R[0,1],2)))

m, b = np.polyfit(df.Precipitation, df.Sea_level, 1)


