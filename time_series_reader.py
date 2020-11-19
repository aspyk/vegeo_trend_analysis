#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:28:50 2019

@author: moparthys
to compile fortran code, mk_fortran.f90:
try first this
f2py -c mk_fortran.f90 -m mankendall_fortran_repeat_exp2 --fcompiler=gfortran

if not try this
LDFLAGS=-shared f2py -c mk_fortran.f90 -m mankendall_fortran_repeat_exp2 --fcompiler=gfortran

dimensions of 4 regions of MSG-Disk

  Character(Len=4), Dimension(1:4) :: zone = (/ 'Euro', 'NAfr', 'SAfr', 'SAme' /)
  Integer,          Dimension(1:4) :: xmin = (/  1550 ,  1240 ,  2140 ,    40  /)
  Integer,          Dimension(1:4) :: xmax = (/  3250 ,  3450 ,  3350 ,   740  /)
  Integer,          Dimension(1:4) :: ymin = (/    50 ,   700 ,  1850 ,  1460  /)
  Integer,          Dimension(1:4) :: ymax = (/   700 ,  1850 ,  3040 ,  2970  /)
"""

from datetime import datetime
import pandas as pd
import numpy as np
import h5py
from netCDF4 import Dataset
import sys
from os import getpid
import re
import glob
import traceback
from timeit import default_timer as timer


def grouper(input_list, n = 2):
    """
    Generate tuple of n consecutive items from input_list
    ex for n=2: [a,b,c,d] -> [[a,b],[b,c],[c,d]]
    """
    for i in range(len(input_list) - (n - 1)):
        yield input_list[i:i+n]


def plot2Darray(v, var='var'):
    """Debug function to plot a 2D array in terminal"""
    import matplotlib.pyplot as plt

    plt.imshow(v)
    imgname = 'h52img_{0}.png'.format(var)
    plt.savefig(imgname)
    print(f'Saved to {imgname}.')
    os.system('/mnt/lfs/d30/vegeo/fransenr/CODES/tools/TerminalImageViewer/src/main/cpp/tiv ' + imgname)


def extract_valid_albedo(file_name,row0,row1,col0,col1):
    """Shortcut function to extract valid albedo data"""
    ## Extract data zone
    try:
        fh5 = h5py.File(file_name,'r')
    except Exception:
        traceback.print_exc()
        print ('File not found, moving to next file, assigning NaN to extacted pixel.')
        return None

    albedo = fh5['AL-BB-DH'][row0:row1,col0:col1] # array of int
    zage = fh5['Z_Age'][row0:row1,col0:col1]
    fh5.close()
    
    ## remove invalid data
    albedo = np.where(albedo==-1, np.nan, albedo/10000.)            
    albedo = np.where(zage>0, np.nan, albedo)
    
    return albedo


def time_series_albedo(start, end, output_path, product_tag, xlim1,xlim2,ylim1,ylim2, nmaster):
    """
    start: datetime object
    end: datetime object
    """
   
    #start = datetime(start_year,start_month,1,0,0,0)
    #end = datetime(end_year,end_month,1,0,0,0)
    series = pd.date_range(start, end, freq='D')

    ## Get the MSG disk land mask (0: see, 1: land, 2: outside space, 3: river/lake)
    hlw = h5py.File('./input_ancillary/HDF5_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_201610171300','r')
    lwmsk = hlw['LWMASK'][xlim1:xlim2,ylim1:ylim2]
    hlw.close()

    # ICARE: 2005-01-01 -> 2016-09-18 (~2005-2016)
    root_dssf = '/cnrm/vegeo/SAT/DATA/AERUS_GEO/Albedo_v104'
    file_paths_root = [root_dssf+'/'+str('{:04}'.format(d.year))+'/' for d in series]
    file_pattern = 'SEV_AERUS-ALBEDO-D3_{}_V1-04.h5'
    file_paths_one = [file_pattern.format(d.strftime('%Y-%m-%d')) for d in series]     
    file_paths_final_icare = [file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
    
    # MDAL: 2004-01-19 -> 2015-12-31 (~2004-2015)
    root_dssf = '/cnrm/vegeo/SAT/DATA/MSG/Reprocessed-on-2017/MDAL'
    file_paths_root = [root_dssf+'/'+str('{:04}'.format(d.year))+'/'+str('{:02}'.format(d.month))+'/'+str('{:02}'.format(d.day))+'/' for d in series]
    file_pattern = 'HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000'
    file_paths_one = [file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final_mdal = [file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
   
    # MDAL_NRT: 2015-11-11 -> today (~2016-today)
    root_dssf = '/cnrm/vegeo/SAT/DATA/MSG/NRT-Operational/AL2'
    file_paths_root_one = root_dssf+'/'+'AL2-{}/'
    #file_pattern = 'HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000.h5' # 2015 -> 2018
    file_pattern = 'HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000' # 2019 -> 2020
    file_paths_one = [file_paths_root_one.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_two = [file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final_mdal_nrt = [file_paths_one[f]+file_paths_two[f] for f in range(len(file_paths_one))]
    
    
    ## the master_chunk or nmaster slices the region and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final = list(range(xlim1, xlim2, nmaster))+[xlim2]
    chunks_col_final = list(range(ylim1, ylim2, nmaster))+[ylim2]
   
    ## grouper() generate pair of (item[n], item[n+1])
    for row0,row1 in grouper(chunks_row_final):
        for col0,col1 in grouper(chunks_col_final):
            t0 = timer()
            print('***', 'Row/Col_SIZE=({},{})'.format(row1-row0, col1-col0), 'GLOBAL_LOC=[{}:{},{}:{}]'.format(row0, row1, col0, col1)) 
           
            ## Initialize an array series with nan
            series_albedo = np.full([len(file_paths_final_icare), row1-row0, col1-col0], np.nan)

            ## Create the chunk mask
            lwmsk_chunk = lwmsk[row0-xlim1:row1-xlim1,col0-ylim1:col1-ylim1]
            ocean = np.where(lwmsk_chunk==0)
            land = np.where(lwmsk_chunk==1)

            for dateindex,date in enumerate(series):
    
                ## Reading ICARE albedo 
                if date.year>=2017:
                    print ("ICARE albedo not available, Using NRT data; year > 2016")
                elif 2005 <= date.year < 2017:     
                    file_icare = file_paths_final_icare[dateindex]
                    print(file_icare)
                    albedo_icare = extract_valid_albedo(file_icare,ro0,row1,col0,col1)
                    ## If error in reading, go to the next iteration
                    if albedo_icare is None:
                        continue
                    
                ## Reading MDAL albedo
                if date.year>=2016:
                    file_mdal = file_paths_final_mdal_nrt[dateindex]
                    if date.year<=2018:
                        file_mdal += '.h5'
                else:   
                    file_mdal = file_paths_final_mdal[dateindex]
                print(file_mdal)
                albedo_mdal = extract_valid_albedo(file_mdal,row0,row1,col0,col1)
                ## If error in reading, go to the next iteration
                if albedo_mdal is None:
                    continue


                ## Apply mask
                net_albedo = np.full((row1-row0,col1-col0), np.nan)

                if 2005 <= date.year <= 2016:
                    if len(ocean[0])>1:
                        net_albedo[ocean] = albedo_icare[ocean]
                    if len(land[0])>1:
                        net_albedo[land] = albedo_mdal[land]
                else:
                    if len(ocean[0])>1:
                        net_albedo[ocean] = albedo_mdal[ocean]
                    if len(land[0])>1:
                        net_albedo[land] = albedo_mdal[land]
                  
                ## Fill series 
                series_albedo[dateindex,:] = net_albedo
             
            print(timer()-t0)
 
            ## Write time series of the data for each master iteration
            write_file = output_path+product_tag+'/store_time_series_'+str(row0)+'_'+str(row1)+'_'+str(col0)+'_'+str(col1)+'.nc'
 
            nc_iter = Dataset(write_file, 'w', format='NETCDF4')
            
            nc_iter.createDimension('x', series_albedo.shape[0])
            nc_iter.createDimension('y', series_albedo.shape[1])
            nc_iter.createDimension('z', series_albedo.shape[2])
            
            var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'), zlib=True)
            nc_iter.variables['time_series_chunk'][:] = series_albedo
                
            nc_iter.close()

            del series_albedo       
 
