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
import pandas as pd
import seaborn as sns
import dask.array as da
from sklearn.metrics import mean_squared_error
from math import sqrt
import os,sys


class SnowTool:
    
    def __init__(self):
        savedir = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_comp_albedo_V1-V2_AVHRR/'


    def extract(self): 
    
        df_path = pd.read_csv('path_to_file.csv', sep=';')

        df_path = df_path.rename(columns = {'Unnamed: 0':'id'})
        df_path = df_path.set_index('id')

        print(df_path)

        ds_batch = xr.open_mfdataset(df_path['path'], parallel=True) #loading ncdf files

        print(ds_batch)

        print("--- Total size (GB):")
        print(ds_batch.nbytes * (2 ** -30)) # get size of the dataset in GB
    
        #getting average albedos over whole time period (used for maps and scatter plots)
        darr = ds_batch['QFLAG'] #getting data for specific band
        print(darr)


        #res = darr.mean(['lon','lat'])
        #res = da.count_nonzero( da.bitwise_and(darr//2**5, 1), ['lon','lat'])
        #res = (darr==32).sum(['lon','lat'])
        #res = xr.ufunc.bitwise_and(darr, 0b100000).sum(['lon','lat'])
        func = lambda x: np.bitwise_and(np.right_shift(x, 5), np.uint64(1))
        func = lambda x: np.bitwise_and(x, np.uint64(1))
        res = xr.apply_ufunc(func, darr, input_core_dims=[['lon','lat']], dask='parallelized', vectorize=True)
        #res = itwise_and(np.right_shift(darr, 5), 1).sum(['lon','lat])
        #res = (darr==32).max(['lon','lat'])
        print(np.array(res))

        sys.exit()

        da_count = ((da>>5)&1) #calculate mean over time
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

if __name__=='__main__':
    s = SnowTool()
    s.extract()
