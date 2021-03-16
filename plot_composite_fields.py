#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:18:25 2020

@author: rantanem
"""


from routines import plotMap, adjust_lons_lats, plot_maxmin_points
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from copy import copy
import sys

# plt.rcParams['figure.figsize'] = (10, 10)


reanalysis = 'era5' # either 'era5' or 'ncep' or 'uerra'

# Select tide gauge
pp='kemi'
# select months
selMonths=[1,2,3,4,5,6,7,8,9,10,11,12]
# selMonths=[9,10,11]
# elevated or high risk
risk='high'
# select events
events = ['compound', 'sea_level', 'prec'] # either 'compound', 'prec', or 'sea_level'
# Title texts
titles = ['a) Compound', 'b) Sea level only', 'c) Precipitation only']



# select color palette for filled contour plot
palette = copy(plt.get_cmap('YlOrRd'))
palette.set_under('white', 1.0) 

# some specifications for contour
kw_clabels = {'fontsize': 10, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}

slp_composites = {}
tcw_composites = {}
N = np.zeros(len(events))

ii=0
for e in events:
    
    Filename = 'Dates_' + e + '_'
    Path = '/home/rantanem/Documents/python/predict/dates/' + pp + '/' 
    path = '/home/rantanem/Documents/python/predict/data/'
    
    filepath = Path + Filename + risk + '_' + pp + '_grid.csv'

    # Get dates
    df = pd.read_csv(filepath)
    df.index = pd.to_datetime(df.Date)
    compDates = df.index


    compDates = compDates[(compDates.year>=1979 ) & (compDates.year<2019 )]
    
    # select those months which were chosen
    cond = sum([compDates.month == m for m in selMonths]).astype(bool)
    compDates = compDates[cond] +  pd.to_timedelta('18H') 
    
    
    dataFile1 = path + 'era5_slp.nc'
    dataFile2 = path + 'era5_tcw.nc'
    ds_slp = xr.open_dataset(dataFile1)
    ds_tcw = xr.open_dataset(dataFile2)
    
    slp_composite = ds_slp.msl.sel(time=compDates).mean(dim='time')/100
    tcw_composite = ds_tcw.tcw.sel(time=compDates).mean(dim='time')
    slp_composites[e] = slp_composite
    tcw_composites[e] = tcw_composite
    N[ii] = len(compDates)
    ii += 1


########## PLOT COMPOSITE ##########################

    # create composite = take average over the dates
    # slp_composite = xr.concat(slp_array,dim='time').mean(dim='time')
    # slp_composite = slp_a / len(compDates)
    # f_composite = f_a / len(compDates)

    # f_composite = xr.concat(f_array,dim='time').mean(dim='time')
    
#Get a new background map figure
fig, ax = plotMap()

ii=0
for e in events:
    
    # # # plot filled field
    f_levels = np.arange(0,32,2)
    # f_levels = np.arange(-8,9,1)
    palette='BrBG'

    f_contour = ax[ii].contourf(ds_tcw.longitude, ds_tcw.latitude, tcw_composites[e], levels = f_levels, zorder=2,  
                            cmap=palette, transform = ccrs.PlateCarree(), 
                            extend='max')
    
    #Plot the sea level pressure contours on the map, in black
    slp_levels = np.arange(880,1140,4)
    slp_contour = ax[ii].contour(ds_slp.longitude, ds_slp.latitude, slp_composites[e], colors='k', 
                                 levels=slp_levels, linewidths=1, zorder=3, 
                                 transform = ccrs.PlateCarree())

    
    # Label SLP contours
    ax[ii].clabel(slp_contour,   np.arange(880,1140,4), **kw_clabels)
    
    # plot low and high pressures
    # plot_maxmin_points(ax[ii], ds_f.lon, ds_f.lat, slp_composite.values, 'min', 100, 
    #                    symbol='L', color='r',  transform=ccrs.PlateCarree())
    # plot_maxmin_points(ax[ii], ds_f.lon, ds_f.lat, slp_composite.values, 'max', 100, 
    #                    symbol='H', color='b',  transform=ccrs.PlateCarree())
    
    # plot titles    
    ax[ii].annotate(titles[ii],(0.0,1.02),xycoords='axes fraction',fontsize=18)   
    
    # plot the number of cases in lower right corner
    at = AnchoredText("N=" + str(int(N[ii])),
                  prop=dict(size=13), frameon=True,
                  loc='lower right'
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[ii].add_artist(at)
    at.zorder = 11
    
    
    ii=ii+1

# add colorbar and its specifications
fig.subplots_adjust(wspace=0.07)
cbar_ax = fig.add_axes([0.2, 0.27, 0.6, 0.03])
cb = fig.colorbar(f_contour, orientation='horizontal',pad=0.05,
                  fraction=0.053, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
cb.set_label(label='Total column water [kg m⁻²]',fontsize=16)


# save figure

figurePath = '/home/rantanem/Documents/python/predict/figures/'
figureName =  pp + '_' + 'pw' + '_' + 'composite' +'_' + reanalysis + '.png'
    
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')