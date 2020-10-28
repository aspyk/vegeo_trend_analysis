#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 01:07:28 2020

@author: moparthys
"""

import numpy as np
from netCDF4 import Dataset
import glob
def merge_time_series (product_tag, xlim1, xlim2, ylim1, ylim2):
   row_chunk_main=np.arange(xlim1,xlim2,500)
   col_chunk_main=np.arange(ylim1,ylim2,500)
   '''we can use 1000 or less file size 500, but need to change the dependencies in code as described by the comments '''
   chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
   chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0) 

   for r in range(len(chunks_row_final)-1):
       for c in range(len(chunks_col_final)-1):
           
           print('store_time_series_{}_{}_{}_{}_.nc'.format(chunks_row_final[r],chunks_row_final[r+1],chunks_col_final[c],chunks_col_final[c+1]))
           file='store_time_series_{}_{}_{}_{}_.nc'.format(chunks_row_final[r],chunks_row_final[r+1],chunks_col_final[c],chunks_col_final[c+1])
           
           
           try:
               list_files_first=glob.glob('./output_timeseries_first/'+product_tag+'/'+file) 
               list_files_next=glob.glob('./output_timeseries_next/'+product_tag+'/'+file)

               print('first_file:', list_files_first[0])
               print('second_file:', list_files_next[0])
               nc1=Dataset('./output_timeseries_first/'+product_tag+'/'+file,'r')
               nc2=Dataset('./output_timeseries_next/'+product_tag+'/'+file,'r')

               nc_iter=Dataset('./output_timeseries_newlist/'+product_tag+'/'+file,'w',format='NETCDF4')

               lents1,xshp1,yshp1=nc1.variables['time_series_chunk'].shape

               lents2,xshp2,yshp2=nc2.variables['time_series_chunk'].shape
               
               lents=lents1+lents2
               
               nc_iter.createDimension('x', lents)
               nc_iter.createDimension('y', xshp1)
               nc_iter.createDimension('z', yshp1)
               
               var1 = nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'),zlib=True)
               nc_iter.variables['time_series_chunk'][0:lents1,:,:]=nc1.variables['time_series_chunk'][:]
               nc_iter.variables['time_series_chunk'][lents1:lents1+lents2,:,:]=nc2.variables['time_series_chunk'][:]

               nc_iter.close()
               nc1.close()
               nc2.close()
               

           except:
               print('No File Found: first_list', list_files_first[0] )
               print('No File Found: second_list', list_files_next[0] )

               pass

           
           
           
           
           
       
       
