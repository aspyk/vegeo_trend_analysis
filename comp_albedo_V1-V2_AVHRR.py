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
import matplotlib.colors as mcolors
import xarray as xr
import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np
import glob
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import os,sys

from tools import SimpleTimer

from dask.distributed import Client

#from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
#from dask.diagnostics import visualize


class V1V2comp:

    def __init__(self):
        self.savedir = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_comp_albedo_V1-V2_AVHRR/'
        self.basedir = '/data/c3s_pdf_live/MTDA/'
        
        #self.folders_to_process = ['C3S_ALBB_BH_Global_4KM', 'C3S_ALBB_DH_Global_4KM', 'C3S_ALSP_BH_Global_4KM', 'C3S_ALSP_DH_Global_4KM']
        self.folders_to_process = ['C3S_ALBB_DH_Global_4KM']
    
    
    ##############################################################################
    def get_albedo(self, datapaths, albedo_type, snow_filtering=False): #load albedos and extract data
    
        DS = xr.open_mfdataset(datapaths[:], parallel=True, chunks='auto') #loading ncdf files
    
        ## Get QFLAG
        ## Give dtype here because lazy loading can't infer it (float by defaut but right_shift requires int)
        da_qflag = DS['QFLAG'].astype(np.uint8)
        da_snowmask = np.logical_and(np.right_shift(da_qflag, 5), 1)==0 # False for snow, True for other

        ## Get albedo data
        da_al = DS[albedo_type] #getting data for specific band
        if snow_filtering:
            da_al = da_al.where(da_snowmask) # filter out snow: set to nan when da_snowmask is False
            if 'snow' not in self.name:
                self.name += '_nosnow'
        #da_mean_lowres = da_al.isel(lat=slice(None, None, 5), lon=slice(None, None, 5)).mean('time') #downsampling for faster plotting
        #da_mean_lowres = da_al.isel(lon=slice(5000, None)).mean('time') #downsampling for faster plotting
        da_mean_lowres = da_al.mean('time') #downsampling for faster plotting
    
        #getting average, min and max albedos for each time step (used to plot timeline)
        da_timeline_mean = da_al.mean(['lon','lat'])
        da_timeline_max  = da_al.max(['lon','lat'])
        da_timeline_min  = da_al.min(['lon','lat'])
    
        res_comp = dd.compute(da_mean_lowres, da_timeline_mean, da_timeline_max, da_timeline_min) 

        return res_comp
    
    ##############################################################################
    def compute_diff(self):
        ti = SimpleTimer('detailed_inner_loop')  # initialize the timer and start counting
        tt = SimpleTimer('total_inner_loop')  # initialize the timer and start counting
        
        if 1:
        #with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:

            #starting loop for albedo type (broadband or spectral, direct or bi-hemispherical)
            for albedo_version in self.folders_to_process:
            
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
                years_to_process = [os.path.basename(x) for x in glob.glob(self.basedir + albedo_version + '_V1/*/*')]
                print(years_to_process)
            
                #starting loop for processing eachavailable year
                #for year in years_to_process:
                for year in years_to_process[:2]:
            
                    self.name = albedo_version + '_' + year 

                    tt()
            
                    print('\n=== PROCESSING YEAR {} ===\n'.format(year))
            
                    #listing files available for V1 and V2
                    print('---', self.basedir + albedo_version + '_V1/*/' + year + '/**/*.nc')
                    datapaths_V1 = glob.glob(self.basedir + albedo_version + '_V1/*/' + year + '/**/*.nc',
                                             recursive=True)  # list all .nc files in directory and subdirectories
                    datapaths_V2 = glob.glob(self.basedir + albedo_version + '_V2/*/' + year + '/**/*.nc',
                                             recursive=True)  # list all .nc files in directory and subdirectories
                    #print('***V1 paths***', datapaths_V1)
                    #print('***V2 paths***', datapaths_V2)
            
                    ti()

                    if 0:
                        self.get_snow([datapaths_V1, datapaths_V2], albedo_version) 
                        sys.exit()
                    
                    #loading albedos
                    albedo_V1, alm1, alx1, aln1 = self.get_albedo(datapaths_V1, albedo_type, snow_filtering=True)
                    albedo_V2, alm2, alx2, aln2 = self.get_albedo(datapaths_V2, albedo_type, snow_filtering=True)
                    #albedo_V1, al_timeline_mean_V1, al_timeline_max_V1, al_timeline_min_V1 = self.get_albedo(datapaths_V1, albedo_type)
                    #albedo_V2, al_timeline_mean_V2, al_timeline_max_V2, al_timeline_min_V2 = self.get_albedo(datapaths_V2, albedo_type)
            
                    ti('   loading albedo')
            
                    #processing diff and stat
                    albedo_diff = albedo_V1 - albedo_V2
                    print(albedo_diff.mean(), albedo_diff.min(), albedo_diff.max())
                    print(albedo_diff.shape)
            
                    #x = albedo_V1.values.flatten() #extract values from data_array
                    #y = albedo_V2.values.flatten() #extract values from data_array
                    x = albedo_V1.values.ravel() #extract values from data_array
                    y = albedo_V2.values.ravel() #extract values from data_array

                    lx = len(x)
                    ly = len(y)
                    print('x_valid_before:', np.count_nonzero(~np.isnan(x))/lx)
                    print('y_valid_before:', np.count_nonzero(~np.isnan(y))/ly)
                    #sys.exit()

                    if 1:
                        ind_nan = np.logical_or(np.isnan(x), np.isnan(y)) #filtering out NaN data
                        x = x[~ind_nan]#filtering out NaN data
                        y = y[~ind_nan]#filtering out NaN data
                        rmse = np.sqrt(np.nanmean((x-y)**2))
                        corr = np.corrcoef(x, y)[0, 1]
                        print('///RMSE///', rmse)
                        print('///corr///', corr)
                        print('x_valid_after:', np.count_nonzero(~np.isnan(x))/lx)
                        print('y_valid_after:', np.count_nonzero(~np.isnan(y))/ly)
                        ti('   processing albedo')
                    else:
                        rmse = -99
                        corr = -99
                        rmse = np.sqrt(np.nanmean((x-y)**2))
                        corr = np.corrcoef(x, y)[0, 1]
            
                    ### TODO
                    # - plot map mean V1 et V2
                    # - zoom on africa
                    # - plot contour on hist2d


            
                    #processing plots
                    print('   --- processing plots')
                    #fig = plt.figure(figsize=(20, 18), dpi=72)  # format A4 paysage : (11.69, 8.27)
                    fig = plt.figure(figsize=(30/2.54, 20/2.54))  # format A4 paysage : (11.69, 8.27)
            
                    #configurating axes layout (1 extra line is kept for future plots upgrades, it could be removed)
                    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3) #
                    ax_diff = fig.add_subplot(grid[0, :2])
                    ax_scatter = fig.add_subplot(grid[0, 2])
                    ax_timeline = fig.add_subplot(grid[1, :3])
            
                    ## Plot diff
                    albedo_diff.plot.imshow(ax=ax_diff, vmin=-.25, vmax=.25, cmap='RdBu_r')
                    #np.isnan(albedo_V1).plot.imshow(ax=ax_diff)
                    ti('   diff_plot')
            
                    ## Plot timeline V1 and V2
                    alm1.plot(ax=ax_timeline, color='b', label='mean albedo V1')
                    alx1.plot(ax=ax_timeline, color='b', linestyle='--')
                    aln1.plot(ax=ax_timeline, color='b', linestyle='--')
                    alm2.plot(ax=ax_timeline, color='r', label='mean albedo V2')
                    alx2.plot(ax=ax_timeline, color='r', linestyle='--')
                    aln2.plot(ax=ax_timeline, color='r', linestyle='--')
                    ax_timeline.legend(loc='upper right')
                    ax_timeline.set_ylim([0, 1])
                    ti('   timeline_plot')
            
                    ## Plot scatter and stats
                    #sns.regplot(x=x, y=y, ax=ax_scatter, scatter_kws={"color": "lightblue", "alpha": 0.3},
                    #            line_kws={"color": "red", "alpha": 0.3})
                    al_max = 0.9
                    #ax_scatter.hist2d(x, y, bins=[np.linspace(0, al_max, 200)]*2, cmin=1, norm=mcolors.LogNorm())
                    counts, xedges, yedges, im = ax_scatter.hist2d(x, y, bins=[np.linspace(0, al_max, 200)]*2, cmin=1, cmap='jet')
                    fig.colorbar(im, ax=ax_scatter)
                    ax_scatter.text(.05, .8, 'RMSE: {:.3f}\nR: {:.3f}'.format(rmse, corr), ha='left', fontsize=8,
                                    bbox={'facecolor': 'white', 'alpha': 0.8,})
                    ax_scatter.plot([0.0, al_max], [0.0, al_max], c='r', lw=1)
                    ax_scatter.set_xlabel('V1')
                    ax_scatter.set_ylabel('V2')
                    ti('   scatter_plot')
            
                    ## Figure tuning
                    fig.suptitle(self.name, size=20, fontweight='bold')
                    ax_diff.set_title('V1 - V2')
                    #ax_scatter.set_title('V2 = f(V1)')
                    ax_scatter.set_aspect('equal', adjustable='box')
                    grid.tight_layout(fig, rect=[0, 0, 1, 0.965], h_pad=0, w_pad=0)  #adjust title position and reduce margins
            
                    #create output directory if it doesn't exist
                    if not os.path.exists(self.savedir + albedo_version):
                        os.makedirs(self.savedir + albedo_version)
            
                    ti('   figure_tuning')
                    #save plot as file
                    plt.savefig(self.savedir + albedo_version + '/' + self.name + '.png', dpi=fig.dpi)
                    # plt.show()
            
                    ti('   save_fig')
            
            
                    ###Initialy it was plan to plot a density scatter plot with the following method, but for some reason it doesn't work (probably too much points to process?)
                    if 0:
                        # #Calculate the point density
                        # xy = np.vstack([x,y])
                        # z = gaussian_kde(xy)(xy)
                        # #Sort the points by density, so that the densest points are plotted last
                        # idx = z.argsort()
                        # ax_scatter.scatter(x, y, c=z, s=50, edgecolor='')
            
                        ###This extra scatter is plotted instead, but sns doesn't allow to easily include it in the first figure...
                        #sns_plot = sns.jointplot(x=albedo_V1.values.flatten(), y=albedo_V2.values.flatten(), kind="hex")
                        sns_plot = sns.jointplot(x=x, y=y, kind="hex")
                        # plt.text(.05, .7, 'RMSE: '+str(rmse)+'\n'+'R: '+str(corr), ha='left', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.1, 'pad': 10})
                        plt.suptitle(self.name + ' V2 = f(V1)', fontsize=15, fontweight='bold')
                        plt.tight_layout()
                        sns_plot.savefig(self.savedir + albedo_version + '/' + self.name + '_scatter.png')
            
                        ti('   scatter_plot #2')
                        
                    tt('{}_{}'.format(albedo_version, year))
        
        ti.show()
        tt.show()

        #visualize([prof, rprof, cprof])
    
    def get_snow(self, datapaths, albedo_version): 

        for datapath in datapaths:
            DS = xr.open_mfdataset(datapath, parallel=True) #loading ncdf files
    
            ## Give the type here because lazy loading can't infer it (float by defaut but right_shift requires int)
            dqflag = DS['QFLAG'].astype(np.uint8)
            da_sum = np.logical_and(np.right_shift(dqflag, 5), 1)==1

            da_sum = da_sum.sum(['lon','lat'])

            res_comp = da_sum.compute() 

            res_comp.plot()
        plt.savefig(self.savedir + albedo_version + '/' + 'test.png')

        sys.exit()



if __name__=='__main__':

    #client = Client()
    #client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
    #client = Client(n_workers=4, threads_per_worker=1)
    #client = Client(n_workers=1, threads_per_worker=4, memory_limit='6GB')
    client = Client(processes=False, memory_limit='7GB')
    #client = Client(memory_limit='1.5GB')
    print(client)

    #dask.config.set(scheduler='threads') # fastest
    #dask.config.set(scheduler='processes') # very long here, because of the GIL ?
    #dask.config.set(scheduler='synchronous')

    proc = V1V2comp()
    proc.compute_diff()
