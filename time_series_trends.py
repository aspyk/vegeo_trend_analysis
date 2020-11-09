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
from scipy import stats
from scipy.stats import mstats
import h5py
import mankendall_fortran_repeat_exp2 as m
from multiprocessing import Pool, cpu_count, RawArray, Lock
import itertools
from netCDF4 import Dataset
import sys
#from kendall_esacci import mk_test
from os import getpid
import re
import glob
''' Prepare your files to read time series''' 


def time_series_albedo(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster):
    start_month=start_month
    end_month=end_month
    start_year=start_year
    end_year=end_year
    
    hlw=h5py.File('./input_ancillary/HDF5_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_201610171300','r')
    lwmsk=hlw['LWMASK'][:]
    

    start = datetime(start_year,start_month,1,0,0,0)
    end = datetime(end_year,end_month,1,0,0,0)
    series=pd.bdate_range(start, end, freq='D')
   
    root_dssf='/cnrm/vegeo/SAT/DATA/AERUS_GEO/Albedo_v104'
    file_paths_root=[root_dssf+'/'+str('{:04}'.format(d.year))+'/' for d in series]
    file_pattern='SEV_AERUS-ALBEDO-D3_{}_V1-04.h5'
    file_paths_one=[file_pattern.format(d.strftime('%Y-%m-%d')) for d in series]     
    file_paths_final_icare=[file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
    
    
    root_dssf='/cnrm/vegeo/SAT/DATA/MSG/Reprocessed-on-2017/MDAL'
    file_paths_root=[root_dssf+'/'+str('{:04}'.format(d.year))+'/'+str('{:02}'.format(d.month))+'/'+str('{:02}'.format(d.day))+'/' for d in series]
    file_pattern='HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000'
    file_paths_one=[file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final_mdal=[file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]
    
    root_dssf='/cnrm/vegeo/SAT/DATA/MSG/NRT-Operational/AL2'
    file_paths_root_one=root_dssf+'/'+'AL2-{}/'
    file_pattern='HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_{}0000.h5'
    file_paths_one=[file_paths_root_one.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_two=[file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    
    file_paths_final_mdal_nrt=[file_paths_one[f]+file_paths_two[f] for f in range(len(file_paths_one))]
    
    
    row_chunk_main=np.arange(xlim1,xlim2,nmaster)
    col_chunk_main=np.arange(ylim1,ylim2,nmaster)
    '''the master_chunk or nmaster above slices the region or box and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)
    
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print(iter_row,iter_col)
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
           
           pandas_series_albedo=np.empty([len(file_paths_final_icare),chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]])
           pandas_series_albedo[:]=np.NaN
           cnt=0             
           for f in range(len(file_paths_final_icare)):
    
               file_iter=file_paths_final_icare[f]
               dayinformation=re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
    
               year=np.int(dayinformation[3])
               month=np.int(dayinformation[4])
               day=np.int(dayinformation[5])
    
               dayindex=np.where(np.logical_and(series.day==day,series.month==month) & np.logical_and(series.year==year,series.month==month))
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
    

def time_series_lai(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster):

    start_month=start_month
    end_month=end_month
    start_year=start_year
    end_year=end_year

    start = datetime(start_year,start_month,1,0,0,0)
    end = datetime(end_year,end_month,1,0,0,0)
    series=pd.bdate_range(start, end, freq='D')

    root_dssf='/cnrm/vegeo/SAT/DATA/MSG_LAI_DAILY_CDR/'
    #file_pattern='HDF5_LSASAF_MSG_LAI-D10_MSG-Disk_{}0000'
    file_pattern='HDF5_LSASAF_MSG_LAI_MSG-Disk_{}0000'
    file_paths_one=[file_pattern.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final=[root_dssf+file_paths_one[f] for f in range(len(file_paths_one))]
    
    row_chunk_main=np.arange(xlim1,xlim2,nmaster)
    col_chunk_main=np.arange(ylim1,ylim2,nmaster)
    '''the master_chunk or nmaster above slices the region or box and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)
    
     
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print(iter_row,iter_col)
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
           
           pandas_series_lai = np.empty([len(file_paths_final),chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]])
           pandas_series_lai[:] = np.NaN
    #       cnt=0             
           for f in range(len(file_paths_final)):
               '''Read time series of the data '''
               try:
                   
                   '''Reading MSG LAI 10 daily ''' 
                   file_iter = file_paths_final[f]
                   dayinformation = re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
                   ymdtag = dayinformation[-1]
    
                   year = np.int(ymdtag[0:4])
                   month = np.int(ymdtag[4:6])
                   day = np.int(ymdtag[6:8])
                   hour = np.int(ymdtag[8:10])
    
                   dayindex = np.where(np.logical_and(series.day==day,series.month==month) & np.logical_and(series.year==year,series.month==month))
    
                   dayindex = np.array(dayindex)
                   rindx,cindx = dayindex.shape
                   print(cindx) 
                   if(cindx==1):
    
                       file = file_iter
                       print(file, iter_row, iter_col)
                       fid = h5py.File(file,'r')
                                     
                       LAI = np.array(fid['LAI'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                       
                       print(LAI.shape)
                       xshape,yshape = LAI.shape[0],LAI.shape[1]
                       LAI = np.reshape(LAI,[xshape*yshape])
                           
                       LAI[LAI==-10] = np.NaN
                       
                       LAI = LAI/1000.     
                       
                       LAI = np.reshape(LAI,[xshape,yshape])
                       
                       
                       
                       pandas_series_lai[dayindex[0][0],:] = LAI
    #                   cnt+=1
                   else:
                        
                        pass
            
               except:
                    print("error")
    #                print('File not found moving to next file, assigning NaN to extacted pixel')
    #                cnt+=1                    
            
                    pass		
    
           '''Write time series of the data for each master iteration '''
           write_file = output_path+product_tag+'/store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'

           nc_iter = Dataset(write_file, 'w', format='NETCDF4')
           
           nc_iter.createDimension('x', pandas_series_lai.shape[0])
           nc_iter.createDimension('y', pandas_series_lai.shape[1])
           nc_iter.createDimension('z', pandas_series_lai.shape[2])
           
           var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'), zlib=True)
           nc_iter.variables['time_series_chunk'][:] = pandas_series_lai
               
           nc_iter.close()
           del pandas_series_lai      
           print(f'Time series file written to: {write_file}')
    

def  time_series_lst(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster):
    
    start_month=start_month
    end_month=end_month
    start_year=start_year
    end_year=end_year

      
    start = datetime(start_year,start_month,1,0,0,0)
    end = datetime(end_year,end_month,1,23,0,0)
    series=pd.bdate_range(start, end, freq='h')

    
    #configfiles1 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/**/*1000', recursive=True)
    #configfiles2 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/**/*1000', recursive=True)
    #configfiles=configfiles1+configfiles2
    
    #configfiles1 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/**/*1100', recursive=True)
    #configfiles2 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/**/*1100', recursive=True)
    #configfiles=configfiles+configfiles1+configfiles2
    
    
    #configfiles1 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/**/*1200', recursive=True)
    #configfiles2 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/**/*1200', recursive=True)
    #configfiles=configfiles+configfiles1+configfiles2
    
    
    
    #configfiles1 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/**/*1300', recursive=True)
    #configfiles2 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/**/*1300', recursive=True)
    #configfiles=configfiles+configfiles1+configfiles2
    
    configfiles1 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/**/*1400', recursive=True)
    configfiles2 = glob.glob('/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/**/*1400', recursive=True)
    #configfiles=configfiles+configfiles1+configfiles2
    configfiles=configfiles1+configfiles2
    
    
    s=series[series.hour==14]
    s=pd.DatetimeIndex.to_series(s)
    
    #s10=series[series.hour==10]
    #s10=pd.DatetimeIndex.to_series(s10)
    
    #s11=series[series.hour==11]
    #s11=pd.DatetimeIndex.to_series(s11)
    #
    #s12=series[series.hour==12]
    #s12=pd.DatetimeIndex.to_series(s12)
    #
    #s13=series[series.hour==13]
    #s13=pd.DatetimeIndex.to_series(s13)
    #
    #s14=series[series.hour==14]
    #s14=pd.DatetimeIndex.to_series(s14)
    
    #s= pd.concat([s10, s11, s12, s13, s14]).sort_index()
    #files_year=[configfiles[f] for f in range(len(configfiles)) if 'MSG-Disk_2017' in configfiles[f]]
    
    
    
    row_chunk_main=np.arange(xlim1,xlim2,nmaster)
    col_chunk_main=np.arange(ylim1,ylim2,nmaster)
    '''the master_chunk or nmaster above slices the region or box and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)
    
    
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print(iter_row,iter_col)
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
           
           pandas_series_lst=np.empty([len(s),chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]])
           pandas_series_lst[:]=np.NaN
           
          
           for f in range(len(configfiles)):
               '''Read time series of the data '''
               try:
                   
                   '''Reading MSG LAI 10 daily ''' 
                   file_iter=configfiles[f]
                   dayinformation=re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
                   ymdtag=dayinformation[-1]
    
                   year=np.int(ymdtag[0:4])
                   month=np.int(ymdtag[4:6])
                   day=np.int(ymdtag[6:8])
                   hour=np.int(ymdtag[8:10])
    
                   dayindex=np.where(np.logical_and(s.index.day==day,s.index.month==month) & np.logical_and(s.index.year==year,s.index.month==month) & np.logical_and(s.index.hour==hour,s.index.minute==0))
    
                   dayindex=np.array(dayindex)
                   rindx,cindx=dayindex.shape
                   print (cindx) 
                   if(cindx==1):
    
                       file=file_iter
                       print(file, iter_row, iter_col)
                       fid=h5py.File(file,'r')
                                     
                       LST_13=np.array(fid['LST'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                       print('LST_13',LST_13.shape)
                       xshape,yshape=LST_13.shape[0],LST_13.shape[1]
                       LST_13=np.reshape(LST_13,[xshape*yshape])
                       LST_13[LST_13==-8000]=np.NaN
    
                       LST_13=np.reshape(LST_13,[xshape,yshape])               
                     
                       pandas_series_lst[dayindex[0][0],:]=LST_13
                       
                   else:
    
                        pass
                    
               except Exception as e:
                   print ('My exception occurred, value:', str(e))
                   pass		
    
    
    
    
           '''Write time series of the data for each master iteration '''
           write_file=output_path+product_tag+'/store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'
           nc_iter = Dataset(write_file, 'w', format='NETCDF4')
           
           nc_iter.createDimension('x', pandas_series_lst.shape[0])
           nc_iter.createDimension('y', pandas_series_lst.shape[1])
           nc_iter.createDimension('z', pandas_series_lst.shape[2])
           
           var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'),zlib=True)
           
           nc_iter.variables['time_series_chunk'][:]=pandas_series_lst
           nc_iter.close()
           del pandas_series_lst             
            
def time_series_evapo(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster):
    
    start_month=start_month
    end_month=end_month
    start_year=start_year
    end_year=end_year

    start = datetime(start_year,start_month,1,0,0,0)
    end = datetime(end_year,end_month,1,0,0,0)
    series=pd.bdate_range(start, end, freq='D')    

    root_lst='/cnrm/vegeo/SAT/DATA/LSA_SAF_METREF_CDR_DAILY/'
    file_pattern_one='HDF5_LSASAF_MSG_METREF_MSG-Disk_{}0000'
    file_paths_one=[file_pattern_one.format(d.strftime('%Y%m%d')) for d in series]     
    file_paths_final=[root_lst+file_paths_one[f] for f in range(len(file_paths_one))]
    
    row_chunk_main=np.arange(xlim1,xlim2,nmaster)
    col_chunk_main=np.arange(ylim1,ylim2,nmaster)
    '''the master_chunk or nmaster above slices the region or box and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)
    
     
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print(iter_row,iter_col)
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
           
           store_series=np.empty([len(file_paths_final),chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]])
           store_series[:]=np.NaN
    #       cnt=0             
           for f in range(len(file_paths_final)):
               '''Read time series of the evapo data '''
               try:
    
                   file_iter=file_paths_final[f]
                   dayinformation=re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
                   ymdtag=dayinformation[-1]
    
                   year=np.int(ymdtag[0:4])
                   month=np.int(ymdtag[4:6])
                   day=np.int(ymdtag[6:8])
                   hour=np.int(ymdtag[8:10])
    
                   dayindex=np.where(np.logical_and(series.day==day,series.month==month) & np.logical_and(series.year==year,series.month==month))
    
                   dayindex=np.array(dayindex)
                   rindx,cindx=dayindex.shape
                   print (cindx) 
                   if(cindx==1):
                       
                       file_iter=file_paths_final[f]
                       dayinformation=re.findall(r"[-+]?\d*\.\d+|\d+", file_iter)
                       print(file_iter)
                       print(file_iter, iter_row, iter_col)
                       fid=h5py.File(file_iter,'r')
                                     
                       EVAPO=np.array(fid['METREF'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                       print('METREF',EVAPO.shape)
                       xshape,yshape=EVAPO.shape[0],EVAPO.shape[1]
                       EVAPO=np.reshape(EVAPO,[xshape*yshape])
                       EVAPO[EVAPO==-8000]=np.NaN
                       EVAPO=EVAPO/100
                       EVAPO=np.reshape(EVAPO,[xshape,yshape])               
    
                       
                       store_series[dayindex[0][0],:]=EVAPO
    #                   cnt+=1
                   else:
                        
                        pass
                       
                       
               except:
                   #                print ('File not found moving to next file, assigning NaN to extacted pixel')
    #                    cnt+=1                    
            
                    pass		
    
           '''Write time series of the data for each master iteration '''
           write_file=output_path+product_tag+'/store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'
           nc_iter = Dataset(write_file, 'w', format='NETCDF4')
           
           nc_iter.createDimension('x', store_series.shape[0])
           nc_iter.createDimension('y', store_series.shape[1])
           nc_iter.createDimension('z', store_series.shape[2])
           
           var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'),zlib=True)
           nc_iter.variables['time_series_chunk'][:]=store_series
               
           nc_iter.close()
           del store_series      

def time_series_dssf(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster):
    
        start_month=start_month
        end_month=end_month
        start_year=start_year
        end_year=end_year
        
        root_dir='/cnrm/vegeo/SAT/DATA/MSG/NRT-Operational/DSSF/'
        start = datetime(start_year,start_month,1,0,0,0)
        end = datetime(end_year,end_month,1,0,0,0)
        series=pd.bdate_range(start, end, freq='30T')
        
        s=series[series.hour==14]
        s=pd.DatetimeIndex.to_series(s)

        root_dssf=root_dir
        file_paths_root=[root_dssf+'DSSF-'+str('{:04}'.format(d.year))+str('{:02}'.format(d.month))+str('{:02}'.format(d.day))+'/' for d in s]
        file_pattern='HDF5_LSASAF_MSG_DSSF_MSG-Disk_{}.h5'
        file_paths_one=[file_pattern.format(d.strftime('%Y%m%d%H%M')) for d in s]     
        file_paths_final=[file_paths_root[f]+file_paths_one[f] for f in range(len(file_paths_one))]


        row_chunk_main=np.arange(xlim1,xlim2,nmaster)
        col_chunk_main=np.arange(ylim1,ylim2,nmaster)
        chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
        chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)

        for iter_row in range(len(chunks_row_final[:]))[0:-1]:
           for iter_col in range(len(chunks_col_final[:]))[0:-1]:
               print(iter_row,iter_col)
               print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
               
               store_series=np.empty([len(file_paths_final),chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]])
               store_series[:]=np.NaN
  
               for f in range(len(file_paths_final)):
                       
                      try:
                          file_dssf=file_paths_final[f]
                    
                          dayinformation=re.findall(r"[-+]?\d*\.\d+|\d+", file_dssf)
                    
                          year=dayinformation[0][0:4]
                          month=dayinformation[0][4:6]
                          day=dayinformation[0][6:8]
                          hhmm=dayinformation[2][8:]
                    
                          hh=hhmm[0:2]
                          mm=hhmm[2:4]
                    
                          day=np.array(day,dtype='int')
                          month=np.array(month,dtype='int')
                          year=np.array(year,dtype='int')
                          hh=np.array(hh,dtype='int')
                          mm=np.array(mm,dtype='int')
                        
                          dayindex=np.where(np.logical_and(s.index.day==day,s.index.month==month) & np.logical_and(s.index.year==year,s.index.month==month) & np.logical_and(s.index.hour==hh,s.index.minute==mm))
                          dayindex=np.array(dayindex)
                          rindx,cindx=dayindex.shape
                          print (cindx) 
                          if(cindx==1):
                              print (file_dssf)
                        
                              file=file_dssf
                              print(file, iter_row, iter_col)
                              
                              fid=h5py.File(file,'r')
                              dssf=np.array(fid['DSSF_Q_Flag'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                              dssf=np.array(fid['DSSF'][chunks_row_final[iter_row]:chunks_row_final[iter_row+1],chunks_col_final[iter_col]:chunks_col_final[iter_col+1]],dtype='f')
                              print('DSSF',dssf.shape)
                              xshape,yshape=dssf.shape[0],dssf.shape[1]
                              dssf=np.reshape(dssf,[xshape*yshape])
                              dssf[dssf==-1]=np.NaN
                              dssf=dssf/10
                              dssf=np.reshape(dssf,[xshape,yshape])               
                              store_series[dayindex[0][0],:]=dssf
                          else:
                              pass
                                       
                                       
                      except:
                              pass		    
                    
           '''Write time series of the data for each master iteration '''
           write_file=output_path+product_tag+'/store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'
           nc_iter = Dataset(write_file, 'w', format='NETCDF4')
           
           nc_iter.createDimension('x', store_series.shape[0])
           nc_iter.createDimension('y', store_series.shape[1])
           nc_iter.createDimension('z', store_series.shape[2])
           
           var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'),zlib=True)
           nc_iter.variables['time_series_chunk'][:]=store_series
               
           nc_iter.close()
           del store_series       
    
