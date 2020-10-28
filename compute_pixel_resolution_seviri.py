#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:46:00 2020

@author: moparthys
"""

import h5py
import numpy as np
from netCDF4 import Dataset

def compute_pixel_resolution (work_dir='./input_ancillary/'):
    # Reading lat from MSG disk
    latmsg=h5py.File(work_dir+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    lat_MSG=latmsg['LAT'][:]
    lat_MSG=np.array(lat_MSG,dtype='float')
    lat_MSG[lat_MSG==910000]=np.nan
    lat_MSG=lat_MSG*0.0001  
    
    lonmsg=h5py.File(work_dir+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG=lonmsg['LON'][:]
    lon_MSG=np.array(lon_MSG,dtype='float')
    lon_MSG[lon_MSG==910000]=np.nan    
    lon_MSG=lon_MSG*0.0001
    
    pixel_map=np.empty([3712,3712])
    pixel_map[:]=np.NaN
    
    for x in np.arange(0,3712,1):
       print (x) 
       one_col_res=np.round(np.diff(lon_MSG[x,:])*112.)
       pixel_map[x,0:-1]=one_col_res
       pixel_map[x,-1]=one_col_res[-2]
       
    nc_output=Dataset('./input_ancillary/SEVIRI_PIX_RESOLUTION.nc','w')
    xdim=nc_output.createDimension('X_dim',3712)
    ydim=nc_output.createDimension('Y_dim',3712)
    Vardim=nc_output.createDimension('Var',1)
    
    output_data=nc_output.createVariable('SEVIRI_PIX_RES',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    
    nc_output.variables['SEVIRI_PIX_RES'][:]=pixel_map
            
    nc_output.close() 
