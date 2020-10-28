#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:47:45 2019

@author: moparthys
"""
#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:28:50 2019

@author: moparthys
to compile:
f2py -c mk_test_final.f90 -m mankendall_fortran_test --fcompiler=gfortran
LDFLAGS=-shared f2py -c mk_test_final_modify.f90 -m mankendall_fortran_repeat_exp_modify_noprint --fcompiler=gfortran

"""


from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import mstats
#import mankendall_fortran_lai as m
from multiprocessing import Pool, Lock
import itertools
from netCDF4 import Dataset
from time import sleep
import h5py
from scipy.spatial import cKDTree
from itertools import product
import sys, glob, os
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o','--output', help='output path')    
    parser.add_argument('-x1','--xlim1', help='limit x1 ')    
    parser.add_argument('-x2','--xlim2', help='limit x2 ')    
    parser.add_argument('-y1','--ylim1', help='limit y1 ')    
    parser.add_argument('-y2','--ylim2', help='limit y2 ')    
    parser.add_argument('-l', '--loglevel', help='log level. CRITICAL ERROR WARNING INFO or DEBUG', default='ERROR')
 
    args = parser.parse_args()
    return args


def init_worker_nc(X_shape,Y_shape,step_chunk_x,step_chunk_y,final_chunk_x,final_chunk_y,start_chunk_x,start_chunk_y,file_pixel_resolution,iteration_final,out_file):

   '''Initialize the pool with boundaries of data to be read and write'''
   '''currently, the last version of the code uses only the input time series data '''
   '''Rest of arguments are not used '''

    # Using a dictionary to initialize all the boundaries of the grid
   var_dict['X_shape'] = X_shape
   var_dict['Y_shape'] = Y_shape
   var_dict['step_chunk_x'] = step_chunk_x
   var_dict['step_chunk_y'] = step_chunk_y
   var_dict['final_chunk_x'] = final_chunk_x
   var_dict['final_chunk_y'] = final_chunk_y
   var_dict['start_chunk_x'] = start_chunk_x
   var_dict['start_chunk_y'] = start_chunk_y
   var_dict['file_input_pixel_resolution'] = file_pixel_resolution
   var_dict['output_paths_chunks_cci']=out_file
#   var_dict['SEVIRI_LAT'] = file_pixel_lat_seviri
#   var_dict['SEVIRI_LON'] = file_pixel_lon_seviri
   var_dict['iteration_last_check'] = iteration_final
 
def compute_weights_projection_seviri(tuple_limits1,tuple_limits2,parent_iteration,child_iteration):

    pix_hdf = Dataset(var_dict['file_input_pixel_resolution'], 'r', format='NETCDF4')
#    pix_hdf=Dataset(file_pixel_resolution,'r')
    work_dir_msg='./input_ancillary/'
    # Reading lat from MSG disk
    hdf_latmsg=h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    
    hdf_lonmsg=h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    
    source_projection_file='./input_ancillary/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004-v2.0.7cds.nc'
    '''please change the cci file here; if you want another year '''
    
    nc_source=Dataset(source_projection_file,'r')
    lat_source=nc_source['lat'][:]
    lon_source=nc_source['lon'][:]
    write_output_path=var_dict['output_paths_chunks_cci']

        
    if tuple_limits2==3700: 
        '''this number will be 3500 instead of 3700 when we process over 500 by 500 sub chunks instead of 100 by 100 '''
        step_chunk2=12
        '''this number will be 212 instead of 12 when we process over 500 by 500 sub chunks instead of 100 by 100 '''
    else: 
        step_chunk2=50
        '''this number will 500 instead of 100 when we process over 500 by 500 sub chunks instead of 100 by 100 '''
  
    if tuple_limits1==3700:
        '''this number will be 3500 instead of 3700 when we process over 500 by 500 sub chunks instead of 100 by 100 '''
        step_chunk1=12
        '''this number will be 212 instead of 12 when we process over 500 by 500 sub chunks instead of 100 by 100 '''
    else:
        step_chunk1=50  
        '''this number will 500 instead of 100 when we process over 500 by 500 sub chunks instead of 100 by 100 '''


    

    '''Create temporary storage with size of sub chunks in main chunk, currently configured 100 by 100 blocks'''
    # var_temp_output=np.empty([step_chunk1,step_chunk2,2,100,2])    
    # var_temp_output[:]=np.NaN
    
    var_output_scores=np.empty([step_chunk1,step_chunk2,2])    
    var_output_scores[:]=np.NaN
    
    var_output_scores_bb=np.empty([step_chunk1,step_chunk2,4])    
    var_output_scores_bb[:]=np.NaN

    var_output_scores_needle=np.empty([step_chunk1,step_chunk2,6])    
    var_output_scores_needle[:]=np.NaN
    
    
    '''NaN matrix by default '''
    print ('pool worker started: ')
    
    print('Blocks #### Row :',tuple_limits1,'TO :',tuple_limits1+step_chunk1,'#### Column :',tuple_limits2,'TO :',tuple_limits2+step_chunk2)

    iter_subchunk_x= tuple_limits1-var_dict['start_chunk_x']
    iter_subchunk_y= tuple_limits2-var_dict['start_chunk_y']
    
    sub_chunks_x=np.arange(tuple_limits1,tuple_limits1+step_chunk1,1)
    sub_chunks_y=np.arange(tuple_limits2,tuple_limits2+step_chunk2,1)

    
    for ii_sub in range(len(sub_chunks_x)):
       for jj_sub in range(len(sub_chunks_y)):
                 pix_res=pix_hdf.variables['SEVIRI_PIX_RES'][sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]

                 lat_MSG_pix=hdf_latmsg['LAT'][sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]
                 lat_MSG_pix=np.array(lat_MSG_pix,dtype='float')
                 lat_MSG_pix[lat_MSG_pix==910000]=np.nan
                 lat_MSG_pix=lat_MSG_pix*0.0001  
                 
                 lon_MSG_pix=hdf_lonmsg['LON'][sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]
                 lon_MSG_pix=np.array(lon_MSG_pix,dtype='float')
                 lon_MSG_pix[lon_MSG_pix==910000]=np.nan    
                 lon_MSG_pix=lon_MSG_pix*0.0001  
                 #print(pix_res,lat_MSG_pix,lon_MSG_pix)
                 if np.isnan(pix_res)!=1:
                     indx_lat=np.where(np.logical_and(lat_source>=lat_MSG_pix-2,lat_source<=lat_MSG_pix+2))
                     indx_lon=np.where(np.logical_and(lon_source>=lon_MSG_pix-2,lon_source<=lon_MSG_pix+2)) 
                     if pix_res>20:
                         pix_res=20
                     if len(indx_lat[0])>1:
                
                         lat_first_index=min(indx_lat[0])
                         lat_last_index=max(indx_lat[0])
                         lon_first_index=min(indx_lon[0])
                         lon_last_index=max(indx_lon[0])
        
                         lon_pos=np.arange(lon_first_index,lon_last_index,1)
                         lat_pos=np.arange(lat_first_index,lat_last_index,1)

                         lat_posmesh,lon_posmesh=np.meshgrid(lat_pos,lon_pos)
                
                         lat_posmesh=np.transpose(lat_posmesh)
                         lon_posmesh=np.transpose(lon_posmesh)
                         
                         latpos_reshp=np.reshape(lat_posmesh,[lat_posmesh.shape[0]*lat_posmesh.shape[1]])
                         lonpos_reshp=np.reshape(lon_posmesh,[lon_posmesh.shape[0]*lon_posmesh.shape[1]])  
                         
                         latlon_posbox=np.array([latpos_reshp,lonpos_reshp])
                         latlon_posbox=np.transpose(latlon_posbox)
                         
                         lon_box=lon_source[lon_first_index:lon_last_index]
                         lat_box=lat_source[lat_first_index:lat_last_index]  
                         lat_mesh,lon_mesh=np.meshgrid(lat_box,lon_box)

                
                         lat_reshp=np.reshape(lat_mesh,[lat_mesh.shape[0]*lat_mesh.shape[1]])
                         lon_reshp=np.reshape(lon_mesh,[lon_mesh.shape[0]*lon_mesh.shape[1]])  

                         latlon_box=np.array([lat_reshp,lon_reshp])
                         latlon_box=np.transpose(latlon_box)               

                         indx_nan=np.where(np.isnan(latlon_box[:,0])!=1)
                                
                         latlon_box_removenan=latlon_box[indx_nan]
                         latlon_posbox_removenan=latlon_posbox[indx_nan]
                    
                         tree = cKDTree(latlon_box_removenan)
                         
                         selection_number_neighbour=np.int(np.round((pix_res/(300/1000))))
                         dists, indexes = tree.query(np.array([lat_MSG_pix,lon_MSG_pix]), k=selection_number_neighbour*selection_number_neighbour)
                         '''Here the selection of the nearest neighbour is based on CKDTree class ; we square the neighbous to complete the box of seviri   '''
                         ''' eg., for pix_res ; 1 km ; seviri pix should hold 9 pixels to complete the box    '''

                         required_positions_nearby=np.array(latlon_posbox_removenan[indexes])     
#                         print('row#',ii_sub,'col#',jj_sub,required_positions_nearby)
                         #var_temp_output[ii_sub,jj_sub,0,0:len(indexes)]=required_positions_nearby

                         #temp_lalon_box=np.array(latlon_box_removenan[indexes])
                         # var_temp_output[ii_sub,jj_sub,1,0:len(indexes)]=temp_lalon_box
                         
                         broad_band_forest=nc_source['lccs_class'][0,required_positions_nearby[:,0],required_positions_nearby[:,1]]
                         broad_band_forest=np.reshape(broad_band_forest,broad_band_forest.shape[0]*broad_band_forest.shape[1])    

                         cat_1=np.where(broad_band_forest==50)
                         cat_2=np.where(broad_band_forest==60)
                         cat_3=np.where(broad_band_forest==61)
                         cat_4=np.where(broad_band_forest==62)

                         sum_cat_bb=0
                         if len(cat_1[0])>1:
                             sum_cat_bb+=len(cat_1[0])
                             var_output_scores_bb[ii_sub,jj_sub,0]=len(cat_1[0])
                             
                         if len(cat_2[0])>1:
                             sum_cat_bb+=len(cat_2[0])
                             var_output_scores_bb[ii_sub,jj_sub,1]=len(cat_2[0])

                         if len(cat_3[0])>1:
                             sum_cat_bb+=len(cat_3[0])
                             var_output_scores_bb[ii_sub,jj_sub,2]=len(cat_3[0])

                         if len(cat_4[0])>1:
                             sum_cat_bb+=len(cat_4[0])
                             var_output_scores_bb[ii_sub,jj_sub,3]=len(cat_4[0])

                         var_output_scores[ii_sub,jj_sub,0]=(sum_cat_bb/len(broad_band_forest))*100

                         cat_1=np.where(broad_band_forest==70)
                         cat_2=np.where(broad_band_forest==71)
                         cat_3=np.where(broad_band_forest==72)
                         cat_4=np.where(broad_band_forest==80)
                         cat_5=np.where(broad_band_forest==81)
                         cat_6=np.where(broad_band_forest==82)

                         sum_cat_needle=0
                         if len(cat_1[0])>1:
                             sum_cat_needle+=len(cat_1[0])
                             var_output_scores_needle[ii_sub,jj_sub,0]=len(cat_1[0])
                             
                         if len(cat_2[0])>1:
                             sum_cat_needle+=len(cat_2[0])
                             var_output_scores_needle[ii_sub,jj_sub,1]=len(cat_2[0])
                             
                         if len(cat_3[0])>1:
                             sum_cat_needle+=len(cat_3[0])
                             var_output_scores_needle[ii_sub,jj_sub,2]=len(cat_3[0])

                         if len(cat_4[0])>1:
                             sum_cat_needle+=len(cat_4[0])
                             var_output_scores_needle[ii_sub,jj_sub,3]=len(cat_4[0])

                         if len(cat_5[0])>1:
                             sum_cat_needle+=len(cat_5[0])
                             var_output_scores_needle[ii_sub,jj_sub,4]=len(cat_5[0])

                         if len(cat_6[0])>1:
                             sum_cat_needle+=len(cat_6[0])
                             var_output_scores_needle[ii_sub,jj_sub,5]=len(cat_6[0])

                         var_output_scores[ii_sub,jj_sub,1]=(sum_cat_needle/len(broad_band_forest))*100
                         

    print('pool completed:  Blocks #### Row :',tuple_limits1,'TO :',tuple_limits1+step_chunk1,'#### Column :',tuple_limits2,'TO :',tuple_limits2+step_chunk2)
    write_string0="ITERATION_MASTER_"+np.str(parent_iteration)+"_ITERATION__CHILD_"+np.str(child_iteration)+"_ROW_"+np.str(tuple_limits1)+'_'+np.str(tuple_limits1+step_chunk1)+'_'+"COL_"+np.str(tuple_limits2)+'_'+np.str(tuple_limits2+step_chunk2)+'_.nc'
    write_string=write_output_path+'/'+write_string0

#    '''I am writing one separate file for each sub process with tag main iteration and sub iteration, I call MASTER and CHILD '''
    if lock.acquire():
        print ('Pool process Thread Locked:')
        print ('Pool process Writing NetCDF:')
        nc_output=Dataset(write_string,'w')
        xdim=nc_output.createDimension('X_dim',step_chunk1)
        ydim=nc_output.createDimension('Y_dim',step_chunk2)
        Vardim=nc_output.createDimension('Var',2)
        Vardim_BB=nc_output.createDimension('Var_BB',4)
        Vardim_Needle=nc_output.createDimension('Var_Needle',6)
        #Pixdim=nc_output.createDimension('pix',100)
        
        '''disabled the storage of positions and latitude longitude to gain speed and memory '''
                     
        #output_data1=nc_output.createVariable('positions',np.float64,('X_dim','Y_dim','pix','Var'),zlib=True,least_significant_digit=3)
        #output_data2=nc_output.createVariable('lat_lon_nearby',np.float64,('X_dim','Y_dim','pix','Var'),zlib=True,least_significant_digit=3)
        output_data3=nc_output.createVariable('scores_bb_cci',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
        output_data4=nc_output.createVariable('scores_needle_cci',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
        output_data5=nc_output.createVariable('bb_cci',np.float64,('X_dim','Y_dim','Var_BB'),zlib=True,least_significant_digit=3)
        output_data6=nc_output.createVariable('needle_cci',np.float64,('X_dim','Y_dim','Var_Needle'),zlib=True,least_significant_digit=3)

        
        #nc_output.variables['positions'][:]=var_temp_output[:,:,0]
        #nc_output.variables['lat_lon_nearby'][:]=var_temp_output[:,:,1]
        nc_output.variables['scores_bb_cci'][:]=var_output_scores[:,:,0]
        nc_output.variables['scores_needle_cci'][:]=var_output_scores[:,:,1]
        nc_output.variables['bb_cci'][:]=var_output_scores_bb
        nc_output.variables['needle_cci'][:]=var_output_scores_needle
        
        
        nc_output.close() 
        lock.release()
        print ('Pool process Thread Released:')
        print ('Pool process worker finished: ')
            
        pix_hdf.close() 
        nc_source.close()
        hdf_latmsg.close()
        hdf_lonmsg.close()
        return None 



def main():
    args = parse_args()

    
#    input_path=args.input

    output_path=args.output
    
    
    xlim1=int(args.xlim1)
    xlim2=int(args.xlim2)
    ylim1=int(args.ylim1)
    ylim2=int(args.ylim2)

    row_chunk_main=np.arange(xlim1,xlim2,500)
    col_chunk_main=np.arange(ylim1,ylim2,500)
    '''we can use 1000 or less file size 500, but need to change the dependencies in code as described by the comments '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)
    
    number_of_main_chunks_row=len(chunks_row_final)
    number_of_main_chunks_col=len(chunks_col_final)


    file_pixel_resolution='./input_ancillary/SEVIRI_PIX_RESOLUTION.nc'    
    '''This SEVIRI pixel Resolution is prepared only once and it will give the pixel resolution at pixel level '''
    
    '''Main block iteration here'''

    ''' Iterating over the main blocks or the master blocks, 500 steps'''
    lock=Lock()   
    main_block_iteration=0
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print(main_block_iteration,iter_row,iter_col)
           '''Write time series of the data for each master iteration '''
           '''calculate trend from the each time series of master chunks''' 
           print('calculating trend for the chunk') 
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
    #       main_block_iteration+=1
    
           X_shape, Y_shape=chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]
           
           '''Creation of sub chunks of 100 by 100 to estimate trends in Master 500 by 500 '''
           '''one can use 500 by 500 sub chunks to estimate trends if Master is by 1000 by 1000 '''
           mainchunks_x=np.arange(chunks_row_final[iter_row],chunks_row_final[iter_row+1],50)
           '''one can use 500 instead of 100 sub chunks to estimate trends if Master is by 1000 by 1000 '''
           mainchunks_y=np.arange(chunks_col_final[iter_col],chunks_col_final[iter_col+1],50)
           '''one can us e 500 instead of 100 sub chunks to estimate trends if Master is by 1000 by 1000 '''
    
           step_chunk_x=chunks_row_final[iter_row+1]-chunks_row_final[iter_row]
           step_chunk_y=chunks_col_final[iter_col+1]-chunks_col_final[iter_col]
           
           final_chunk_x=chunks_row_final[iter_row+1]
           final_chunk_y=chunks_col_final[iter_col+1]
           
           start_chunk_x=chunks_row_final[iter_row]
           start_chunk_y=chunks_col_final[iter_col]
          
           inputdata=[mainchunks_x,mainchunks_y]
           result_main_chunks = list(itertools.product(*inputdata))
           result_chunks=[(result_main_chunks[i][0],result_main_chunks[i][1],main_block_iteration,i) for i in range(len(result_main_chunks))]
           print (result_chunks)
           iteration_final=len(result_chunks)-1
    
           ''' Applying the multiple processing here, with process of choice, i use 4 for local and 16 for lustre '''
           ''' Here we pass the arguments to the initializer for pool so we use in the function used in multiple processing, it can be changed differently '''
           with Pool(processes=16, initializer=init_worker_nc, initargs=(X_shape,Y_shape,step_chunk_x,step_chunk_y,final_chunk_x,final_chunk_y,start_chunk_x,start_chunk_y,file_pixel_resolution,iteration_final,output_path)) as pool:
               '''I am not returning results, usually you can return and write the results after multiprocessing '''
               print('pool started')
               results = pool.starmap(compute_weights_projection_seviri,[(result_chunks[i]) for i in range(len(result_chunks))])
               pool.close()
               pool.join()
               print('pool cleaned')
           '''iterate the main block iteration'''
           main_block_iteration+=1
    
    
    
    
if __name__ == "__main__":
    try:
        """ runs the concatenation of timeserie of albedo, lai and fapar"""
        lock = Lock()
        var_dict = {}

        '''Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main '''
        main()
    except :
        sys.exit()
