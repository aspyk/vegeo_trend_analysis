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
''' Prepare your files to read time series''' 

def grouper(input_list, n = 2):
    for i in range(len(input_list) - (n - 1)):
        yield input_list[i:i+n]

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
    lwmsk = hlw['LWMASK'][:]

   
    root_dssf = '/cnrm/vegeo/SAT/DATA/AERUS_GEO/Albedo_v104'
    file_paths_root = [root_dssf+'/'+str('{:04}'.format(d.year))+'/' for d in series]
    file_pattern = 'SEV_AERUS-ALBEDO-D3_{}_V1-04.h5'
    file_paths_one = [file_pattern.format(d.strftime('%Y-%m-%d')) for d in series]     
    file_paths_final_icare = [file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
    
    root_dssf = '/cnrm/vegeo/SAT/DATA/MSG/Reprocessed-on-2017/MDAL'
    file_paths_root = [root_dssf+'/'+str('{:04}'.format(d.year))+'/'+str('{:02}'.format(d.month))+'/'+str('{:02}'.format(d.day))+'/' for d in series]
    file_pattern = 'HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000'
    file_paths_one = [file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final_mdal = [file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
    
    root_dssf = '/cnrm/vegeo/SAT/DATA/MSG/NRT-Operational/AL2'
    file_paths_root_one = root_dssf+'/'+'AL2-{}/'
    file_pattern = 'HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000.h5'
    file_paths_one = [file_paths_root_one.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_two = [file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final_mdal_nrt = [file_paths_one[f]+file_paths_two[f] for f in range(len(file_paths_one))]
    
    
    ## the master_chunk or nmaster slices the region and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final = list(range(xlim1, xlim2, nmaster))+[xlim2]
    chunks_col_final = list(range(ylim1, ylim2, nmaster))+[ylim2]
   
    ## grouper() generate pair of (item[n], item[n+1])
    for row0,row1 in grouper(chunks_row_final):
    #for iter_row in range(len(chunks_row_final)-1):
       for col0,col1 in grouper(chunks_col_final):
       #for iter_col in range(len(chunks_col_final)-1):
           print("chunk row {}:{} - col {}:{}".format(row0, row1, col0, col1))
           print(row0, row1, col0, col1, '***Row_SIZE***', row1-row0, '***Col_SIZE***', col1-col0) 
           continue
           
           pandas_series_albedo = np.empty([len(file_paths_final_icare), row1-row0, col1-col0])
           pandas_series_albedo[:] = np.NaN
           cnt = 0             
           for f in range(len(file_paths_final_icare)):
    
               file_iter = file_paths_final_icare[f]
               dayinformation = re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
    
               year  = int(dayinformation[3])
               month = int(dayinformation[4])
               day   = int(dayinformation[5])
    
               dayindex = np.where(np.logical_and(series.day==day,series.month==month) & np.logical_and(series.year==year,series.month==month))
               
               '''Read time series of the data '''
               try:
                   
                   '''Reading ICARE albedo ''' 
    
                   if series[f].year>=2017:
                       print ("ICARE albedo not found for Ocean, Using NRT data; year > 2016")
                   elif series[f].year>=2005 and series[f].year<2017 :     
                       file_icare=file_paths_final_icare[f]
                       fid_icare=h5py.File(file_icare,'r')
                       print (file_icare, iter_row, iter_col)
                                     
    
                       albedo_icare=np.array(fid_icare['ALBEDO']['AL-BB-DH'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                       
                       print (albedo_icare.shape)
                       xshape,yshape=albedo_icare.shape[0],albedo_icare.shape[1]
                       albedo_icare=np.reshape(albedo_icare,[xshape*yshape])
                           
                       albedo_icare[albedo_icare==-1]=np.NaN
                       
                       albedo_icare=albedo_icare/10000.            
                       
                       zage_icare=np.array(fid_icare['ALBEDO']['Z_Age'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]])
                       
                       zage_icare=np.reshape(zage_icare,[xshape*yshape])
                       
                       fid_icare.close()
                       
                       
                       
                       invalid_zage=np.where(zage_icare!=0)
                       if len(invalid_zage[0])>1:
                           albedo_icare[invalid_zage[0]]=np.NaN
                       
                       albedo_icare=np.reshape(albedo_icare,[xshape*yshape])    
         
    
                   '''Reading MDAL albedo ''' 
                   if series[f].year>=2016:
                        file_mdal=file_paths_final_mdal_nrt[f]
                        fid_mdal=h5py.File(file_mdal,'r')
                        print (file_mdal, iter_row, iter_col)
                   else:   
                       file_mdal=file_paths_final_mdal[f]
                       fid_mdal=h5py.File(file_mdal,'r')
                       print(file_mdal, iter_row, iter_col)
    
                   albedo_mdal=np.array(fid_mdal['AL-BB-DH'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                   print (albedo_mdal.shape)
                   xshape,yshape=albedo_mdal.shape[0],albedo_mdal.shape[1]
                   albedo_mdal=np.reshape(albedo_mdal,[xshape*yshape])
                       
                   albedo_mdal[albedo_mdal==-1]=np.NaN
                   
                   albedo_mdal=albedo_mdal/10000.            
                   
                   zage_mdal=np.array(fid_mdal['Z_Age'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]])
                   
                   zage_mdal=np.reshape(zage_mdal,[xshape*yshape])
                   
                   fid_mdal.close()
                   
                   invalid_zage=np.where(zage_mdal!=0)
                   if len(invalid_zage[0])>1:
                       albedo_mdal[invalid_zage[0]]=np.NaN
    
                   net_albedo=np.empty([xshape*yshape])
                   net_albedo[:]=np.NaN
                   
                   albedo_mdal=np.reshape(albedo_mdal,[xshape*yshape])    
    
                   lwmsk_chunk=lwmsk[chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]]
                   lwmsk_chunk=np.reshape(lwmsk_chunk,[xshape*yshape])
                   find_ocean=np.where(lwmsk_chunk==0)
                   find_land=np.where(lwmsk_chunk==1)
                   if series[f].year>=2005 and series[f].year<2017:
                       if len(find_ocean[0])>1:
                           net_albedo[find_ocean[0]]=albedo_icare[find_ocean[0]]
                       if len(find_land[0])>1:
                           net_albedo[find_land[0]]=albedo_mdal[find_land[0]]
                   else:
                       if len(find_ocean[0])>1:
                           net_albedo[find_ocean[0]]=albedo_mdal[find_ocean[0]]
                       if len(find_land[0])>1:
                           net_albedo[find_land[0]]=albedo_mdal[find_land[0]]
                     
                   
                   net_albedo=np.reshape(net_albedo,[xshape,yshape])
     
                   
                   pandas_series_albedo[dayindex[0],:]=net_albedo
                   cnt+=1
            
               except:
                    print ('File not found moving to next file, assigning NaN to extacted pixel')
                    cnt+=1                    
            
                    pass		
    
           '''Write time series of the data for each master iteration '''
           write_file=output_path+product_tag+'/store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'

    
           nc_iter = Dataset(write_file, 'w', format='NETCDF4')
           
           nc_iter.createDimension('x', pandas_series_albedo.shape[0])
           nc_iter.createDimension('y', pandas_series_albedo.shape[1])
           nc_iter.createDimension('z', pandas_series_albedo.shape[2])
           
           var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'),zlib=True)
           nc_iter.variables['time_series_chunk'][:]=pandas_series_albedo
               
           nc_iter.close()
           del pandas_series_albedo       
    
