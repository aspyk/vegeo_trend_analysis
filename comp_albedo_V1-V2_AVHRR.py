#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
This code allows to compare C3S albedos V1 and V2 year by year by plotting difference maps, scatter plots and timeline

TO DO:
-Overall plots and stats (for the whole period) are currently processed by another script (comp_V1-V2_AVHRR_Overall.py) but it could be calculated here by summing data from each year and dividing by number of years
-For spectral albedo, only MIR data is processed, this needs to be updated (looping over spectral bands in the nc file)
-QFLAG should be used to filter out snowy pixels, but a problem was encountered while reading QFLAG with xarray, there are read as NaN even by forcing type as unint8. This has to be fixed.
-This code shoud be adapted to process VGT data. But data resolution of 1km (VS 4km for AVHRR) leads to memory issues. Chunking/slicing should be used to solve this issue.
"""


import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import os,sys

savedir = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_comp_albedo_V1-V2_AVHRR/'
basedir = '/data/c3s_pdf_live/MTDA/'

folders_to_process = ['C3S_ALBB_BH_Global_4KM', 'C3S_ALBB_DH_Global_4KM', 'C3S_ALSP_BH_Global_4KM', 'C3S_ALSP_DH_Global_4KM']


##############################################################################
def get_albedo(datapaths, albedo_type): #load albedos and extract data

    DS = xr.open_mfdataset(datapaths[:], parallel=True) #loading ncdf files

    #getting average albedos over whole time period (used for maps and scatter plots)
    da = DS[albedo_type] #getting data for specific band
    da_mean = da.mean('time') #calculate mean over time
    #da_mean_lowres = da_mean.sel(lat=slice(70, 30)).sel(lon=slice(-25, 70)) # this can be used to zoom in over Europe
    da_mean_lowres = da_mean.isel(lat=slice(None, None, 10)).isel(lon=slice(None, None, 10)) #downsampling for faster plotting

    #getting average, min and max albedos for each time step (used to plot timeline)
    da_timeline_mean = da.mean(['lon','lat'])
    da_timeline_max = da.max(['lon','lat'])
    da_timeline_min = da.min(['lon','lat'])

    #closing arrays to free memory
    DS.close()
    da.close()
    da_mean.close()

    return da_mean_lowres, da_timeline_mean, da_timeline_max, da_timeline_min

    da_mean_lowres.close()
    da_timeline_mean.close()
    da_timeline_max.close()
    da_timeline_min.close()
##############################################################################

#starting loop for albedo type (broadband or spectral, direct or bi-hemispherical)
for albedo_version in folders_to_process:

    print(albedo_version)

    #routine for converting band name from folder name
    #to_do: currently only MIR is read, all bandwidths should be processed
    albedo_type = 'AL_' + albedo_version[9:11] + '_' + albedo_version[6:8] #plot toutes des bandes du nc?
    if albedo_type=='AL_BH_SP':
        albedo_type = 'AL_BH_MIR'
    if albedo_type=='AL_DH_SP':
        albedo_type = 'AL_DH_MIR'
    print(albedo_type)

    #determine years available in the data directory
    years_to_process = [os.path.basename(x) for x in glob.glob(basedir + albedo_version + '_V1/*/*')]
    print(years_to_process)

    #starting loop for processing eachavailable year
    #for year in years_to_process:
    for year in years_to_process[:1]:

        print('***processing year ' + year)

        #listing files available for V1 and V2
        print('---', basedir + albedo_version + '_V1/*/' + year + '/**/*.nc')
        datapaths_V1 = glob.glob(basedir + albedo_version + '_V1/*/' + year + '/**/*.nc',
                                 recursive=True)  # list all .nc files in directory and subdirectories
        datapaths_V2 = glob.glob(basedir + albedo_version + '_V2/*/' + year + '/**/*.nc',
                                 recursive=True)  # list all .nc files in directory and subdirectories
        print('***V1 paths***', datapaths_V1)
        print('***V2 paths***', datapaths_V2)

        #loading albedos
        albedo_V1, al_timeline_mean_V1, al_timeline_max_V1, al_timeline_min_V1 = get_albedo(datapaths_V1, albedo_type)
        albedo_V2, al_timeline_mean_V2, al_timeline_max_V2, al_timeline_min_V2 = get_albedo(datapaths_V2, albedo_type)

        #processing diff and stat
        albedo_diff = albedo_V1 - albedo_V2

        x = albedo_V1.values.flatten() #extract values from data_array
        y = albedo_V2.values.flatten() #extract values from data_array
        ind_nan = np.logical_or(np.isnan(x), np.isnan(y)) #filtering out NaN data
        x = x[np.where(~ind_nan)]#filtering out NaN data
        y = y[np.where(~ind_nan)]#filtering out NaN data
        rmse = round(sqrt(mean_squared_error(x, y)), 2)
        corr = round(np.corrcoef(x, y)[0, 1], 2)
        print('///RMSE///', rmse)
        print('///corr///', corr)


        #processing plots
        print('  -processing plots')
        fig = plt.figure(figsize=(20, 18), dpi=72)  # format A4 paysage : (11.69, 8.27)

        #configurating axes layout (1 extra line is kept for future plots upgrades, it could be removed)
        grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3) #
        ax_diff = fig.add_subplot(grid[0, :2])
        ax_scatter = fig.add_subplot(grid[0, 2])
        ax_timeline = fig.add_subplot(grid[2, :3])

        #plot diff
        albedo_diff.plot(ax=ax_diff, vmin=-.8, vmax=.8, cmap='RdBu_r')

        #plot timeline V1 and V2
        al_timeline_mean_V1.plot(ax=ax_timeline, color='b', label='mean albedo V1')
        al_timeline_max_V1.plot(ax=ax_timeline, color='b', linestyle='--')
        al_timeline_min_V1.plot(ax=ax_timeline, color='b', linestyle='--')
        al_timeline_mean_V2.plot(ax=ax_timeline, color='r', label='mean albedo V2')
        al_timeline_max_V2.plot(ax=ax_timeline, color='r', linestyle='--')
        al_timeline_min_V2.plot(ax=ax_timeline, color='r', linestyle='--')
        ax_timeline.legend(loc='upper right')
        ax_timeline.set_ylim([0, 1])

        #plot scatter and stats
        sns.regplot(x=x, y=y, ax=ax_scatter, scatter_kws={"color": "lightblue", "alpha": 0.3},
                    line_kws={"color": "red", "alpha": 0.3})
        ax_scatter.text(.05, .7, 'RMSE: ' + str(rmse) + '\n' + 'R: ' + str(corr), ha='left', fontsize=12,
                        bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})
        name = albedo_version + ' [' + year + ']'

        #figure tuning
        fig.suptitle(name, size=20, fontweight='bold')
        ax_diff.set_title('V1 - V2')
        ax_scatter.set_title('V2 = f(V1)')
        ax_scatter.set_aspect('equal', adjustable='box')
        grid.tight_layout(fig, rect=[0, 0, 1, 0.965], h_pad=0, w_pad=0)  #adjust title position and reduce margins

        #create output directory if it doesn't exist
        if not os.path.exists(savedir + albedo_version):
            os.makedirs(savedir + albedo_version)

        #save plot as file
        plt.savefig(savedir + albedo_version + '/' + name + '.png', dpi=fig.dpi)
        # plt.show()



        ###Initialy it was plan to plot a density scatter plot with the following method, but for some reason it doesn't work (probably too much points to process?)
        # #Calculate the point density
        # xy = np.vstack([x,y])
        # z = gaussian_kde(xy)(xy)
        # #Sort the points by density, so that the densest points are plotted last
        # idx = z.argsort()
        # ax_scatter.scatter(x, y, c=z, s=50, edgecolor='')

        ###This extra scatter is plotted instead, but sns doesn't allow to easily include it in the first figure...
        sns_plot = sns.jointplot(x=albedo_V1.values.flatten(), y=albedo_V2.values.flatten(), kind="hex")
        # plt.text(.05, .7, 'RMSE: '+str(rmse)+'\n'+'R: '+str(corr), ha='left', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})
        plt.suptitle(name + ' V2 = f(V1)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        sns_plot.savefig(savedir + albedo_version + '/' + name + '_scatter.png')
