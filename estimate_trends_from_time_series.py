"""
Created on Fri Oct  4 09:47:45 2019

@author: moparthys

code to prepare tendencies based on Man-kendall test

"""


from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import mstats
import mankendall_fortran_repeat_exp2 as m
from multiprocessing import Pool, Lock, current_process
import itertools
from netCDF4 import Dataset
from time import sleep
import sys, glob, os
import argparse
import traceback
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import psutil


def parse_args():
    parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-styr','--start_year', help='start year for read of time series')    
    parser.add_argument('-enyr','--end_year', help='end year for read of time series')    
    parser.add_argument('-stmn','--start_month', help='start month for read of time series')    
    parser.add_argument('-enmn','--end_month', help='end month for read of time series')    
    parser.add_argument('-i','--input', help='input path')    
    parser.add_argument('-o','--output', help='output path')    
    parser.add_argument('-ptag','--product_tag', help='product tag or product dataset, either albedo, lai, evapo, dssf, fapar ')    
    parser.add_argument('-x1','--xlim1', help='limit x1 ')    
    parser.add_argument('-x2','--xlim2', help='limit x2 ')    
    parser.add_argument('-y1','--ylim1', help='limit y1 ')    
    parser.add_argument('-y2','--ylim2', help='limit y2 ')    
    parser.add_argument('-n_master','--master_chunk', help='size of master chunks')
    
    parser.add_argument('-c','--choice', help='product tag or product dataset, either ALBEDO, LAI, EVAPO, DSSF, FAPAR ')    
    parser.add_argument('-l', '--loglevel', help='log level. CRITICAL ERROR WARNING INFO or DEBUG', default='ERROR')
    args = parser.parse_args()
    return args

    
def init_worker_nc(var_dict,X_shape,Y_shape,step_chunk_x,step_chunk_y,final_chunk_x,final_chunk_y,start_chunk_x,start_chunk_y,file_timeseries,file_outpath,iteration_final,indx_filter):
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
        var_dict['Indx_seasons'] = indx_filter
        var_dict['file_input_time_series'] = file_timeseries
        var_dict['file_output_tendencies'] = file_outpath
        var_dict['iteration_last_check'] = iteration_final
    
 
def processInput_trends(tuple_limits1,tuple_limits2,parent_iteration,child_iteration, nchild):
    '''This is the main file that calculate trends '''

    process = psutil.Process(os.getpid())
    current = current_process()
    print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')

    '''Reading the input time series file from main chunk, configured length of time series X 500 X 500; it varies if different chunks are used'''
    hdf_ts = Dataset(var_dict['file_input_time_series'], 'r', format='NETCDF4')
    
    '''the tuple limits are between 0 and 3700 for LSA SAF products. if it reaches the limit, it is used for 12 additional pixels to complete size 3712'''
    '''tuple_limits2 and tuple_limits1 correspond to first indices of the bounding box in the iteration'''
    '''currently nchild or child chunks are set to 100 X 100 only, irrespective of master size; eg., 500 X 500 box contain 25 100 X 100 child chunks '''
    if tuple_limits2==3700: 
        step_chunk2=12
    else: 
        step_chunk2=nchild
  
    if tuple_limits1==3700:
        step_chunk1=12
    else:
        step_chunk1=nchild



    '''Create temporary storage with size of sub chunks in main chunk, currently configured 100 by 100 blocks'''
    var_temp_output=np.empty([step_chunk1,step_chunk2,4])    
    var_temp_output[:]=np.NaN
    '''NaN matrix by default '''
    print ('pool worker started: ')
    
    print('Blocks #### Row :',tuple_limits1,'TO :',tuple_limits1+step_chunk1,'#### Column :',tuple_limits2,'TO :',tuple_limits2+step_chunk2)

    iter_subchunk_x= tuple_limits1-var_dict['start_chunk_x']
    iter_subchunk_y= tuple_limits2-var_dict['start_chunk_y']
    

    sub_chunks_x=np.arange(iter_subchunk_x,iter_subchunk_x+step_chunk1,1)
    sub_chunks_y=np.arange(iter_subchunk_y,iter_subchunk_y+step_chunk2,1)

    print(hdf_ts.variables['time_series_chunk'].shape)

    b_deb = 0 # flag to print time profiling
    t00 = timer()
    t000 = timer()
    t_mean = 0.
    print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')
    for ii_sub in range(len(sub_chunks_x)):
        # dimension of variable: time,x,y
        # preload all the y data here to avoid overhead due to calling Dataset.variables at each iteration in the inner loop
        data_test0 = hdf_ts.variables['time_series_chunk'][:,sub_chunks_x[ii_sub],:]
        #data_test0 = hdf_ts.variables['time_series_chunk'][:500,sub_chunks_x[ii_sub],:]
        for jj_sub in range(len(sub_chunks_y)):
            if b_deb: print('---------------')
            if b_deb: print(f'{ii_sub} ({sub_chunks_x[0]},{sub_chunks_x[-1]})', f'{jj_sub} ({sub_chunks_y[0]},{sub_chunks_y[-1]})')
            
            t0 = timer()

            data_test = data_test0[:,sub_chunks_y[jj_sub]]
            #data_test=hdf_ts.variables['time_series_chunk'][:,sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]
            slope=999.0

            if b_deb:
                print('t0', timer()-t0)
                t0 = timer()

            if b_deb: print('Data:', f'{data_test.size - np.isnan(data_test).sum()}/{data_test.size}')
            
            if 0:
            #if (data_test.size - np.isnan(data_test).sum())>0: # test if there are not only NaNs (12 threshold has been removed here)
            #if (data_test.size - np.isnan(data_test).sum())>12: # test if there is at least 12 numbers (why 12 here ?...)
            #if len(np.where(np.isnan(data_test)!=1)[0])>12: # previous version 3 ot 4 times slower

                #print('Use mstats')

                data_sen=np.ma.masked_array(data_test, mask=np.isnan(data_test))

                t0 = timer()

                slope, intercept, lo_slope, up_slope = mstats.theilslopes(data_sen, alpha=0.1)
                
                if b_deb: print('t02', timer()-t0)
                t0 = timer()

            ''' this mstats give correct slope and is consistent with python man-kendall score of Sn; this is fast than Fortran'''

            ''' stats.theilslopes is giving incorrect values when NaN are inside data'''
            
            if b_deb:
                print('t2', timer()-t0)
                t0 = timer()

            data_test[np.isnan(data_test)==1]=999  
            
            if b_deb:
                print('t3', timer()-t0)
                t0 = timer()

            '''we put 999 for missing values so as to validate with fortran code; man-kendall scores are estimated by a fortran file
            already imported'''
            p,z,Sn,nx = m.mk_trend(len(data_test),np.arange(len(data_test)),data_test)
            #p,z,Sn,nx = [0,0,0,0] 

            if b_deb:
                print('t4', timer()-t0)
                t0 = timer()

            if b_deb: print('p,z,slope,nx', p,z,slope,nx)

            var_temp_output[ii_sub,jj_sub,0] = p
            var_temp_output[ii_sub,jj_sub,1] = z
            #var_temp_output[ii_sub,jj_sub,2] = slope
            var_temp_output[ii_sub,jj_sub,2] = Sn
            var_temp_output[ii_sub,jj_sub,3] = nx

        t_mean += timer()-t00
        #print(f't00 {ii_sub} {t_mean/(ii_sub+1)}')
        print(f't00.p{current._identity[0]}.it{ii_sub} {t_mean/(ii_sub+1):.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
        t00 = timer()
    
    print(f'ttot.p{current._identity[0]} {timer()-t000:.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
    print('pool completed:  Blocks #### Row :',tuple_limits1,'TO :',tuple_limits1+step_chunk1,'#### Column :',tuple_limits2,'TO :',tuple_limits2+step_chunk2)
    write_string0 = ("ITERATION_MASTER_" + np.str(parent_iteration)  
                     + "_ITERATION_CHILD_" + np.str(child_iteration)  
                     + "_ROW_" + np.str(tuple_limits1) + '_' + np.str(tuple_limits1+step_chunk1) 
                     + "_COL_" + np.str(tuple_limits2) + '_' + np.str(tuple_limits2+step_chunk2) 
                     + '_.nc')
    write_string = var_dict['file_output_tendencies'] + '/' + write_string0
    write_string = os.path.normpath(write_string)
    '''var_dict['file_output_tendencies'] prepared from the initializer routes the tendencies files to output destination as defined by product_tag previously '''

    if lock.acquire():
        print ('Pool process Thread Locked:')
        print ('Pool process Writing NetCDF:')
        nc_output = Dataset(write_string,'w')
        xdim = nc_output.createDimension('X_dim',step_chunk1)
        ydim = nc_output.createDimension('Y_dim',step_chunk2)
        Vardim = nc_output.createDimension('Var',4)
        
                     
        output_data = nc_output.createVariable('chunk_scores_p_val',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
        output_data = nc_output.createVariable('chunk_scores_z_val',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
        output_data = nc_output.createVariable('chunk_scores_Sn_val',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=8)
        output_data = nc_output.createVariable('chunk_scores_length',np.float64,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
        
        nc_output.variables['chunk_scores_p_val'][:]=var_temp_output[:,:,0]
        nc_output.variables['chunk_scores_z_val'][:]=var_temp_output[:,:,1]
        nc_output.variables['chunk_scores_Sn_val'][:]=var_temp_output[:,:,2]
        nc_output.variables['chunk_scores_length'][:]=var_temp_output[:,:,3]
        
        nc_output.close() 
        lock.release()
        print ('Pool process Thread Released:')
        print ('Pool process worker finished: ')
   
        hdf_ts.close() 
        return None 

def main():
    args = parse_args()

    start_year=int(args.start_year)
    end_year=int(args.end_year)
    
    start_month=int(args.start_month)
    end_month=int(args.end_month)
    
    input_path=args.input

    output_path=args.output
    
    product_tag=args.product_tag
    
    xlim1=int(args.xlim1)
    xlim2=int(args.xlim2)
    ylim1=int(args.ylim1)
    ylim2=int(args.ylim2)
    nmaster=int(args.master_chunk)
    
    choice=args.choice
    
    if choice=='ALBEDO' and product_tag=='albedo':
        print('ALBEDO process')
#        inpath_final=input_path+'/'
        inpath_final=input_path+'/'+product_tag+'/'
        outpath_final=output_path+'/'+product_tag+'/'
    if choice=='LST' and product_tag=='lst':
        print('LST process')
        inpath_final=input_path+'/'+product_tag+'/'
        outpath_final=output_path+'/'+product_tag+'/'
    if choice=='LAI' and  product_tag=='lai':
        print('LAI process')
#        inpath_final=input_path+'/'
        inpath_final=input_path+'/'+product_tag+'/'
        outpath_final=output_path+'/'+product_tag+'/'
    if choice=='EVAPO' and product_tag=='evapo':
        print('EVAPO process')
        inpath_final=input_path+'/'+product_tag+'/'
        outpath_final=output_path+'/'+product_tag+'/'
    if choice=='DSSF' and product_tag=='dssf':
        print('DSSF process')
#        inpath_final=input_path+'/'
        inpath_final=input_path+'/'+product_tag+'/'
        outpath_final=output_path+'/'+product_tag+'/'

    start = datetime(start_year,start_month,1,0,0,0)
    end = datetime(end_year,end_month,1,0,0,0)
    series=pd.bdate_range(start, end, freq='D')
    ''' one can select trends only for a particular season here; currently hard coded; can be given as arguments if wanted'''
    indx_seasons=np.where(np.logical_or(np.logical_or(series.month==6,series.month==7), np.logical_or(series.month==7,series.month==8)))
    
    row_chunk_main=np.arange(xlim1,xlim2,nmaster)
    col_chunk_main=np.arange(ylim1,ylim2,nmaster)
    '''currently tested to 500 X 500 size chunks, but can be changed to other master chunks 100, 200 etc., not less than 100 or in between two integrals '''
    chunks_row_final=np.append(row_chunk_main,[xlim2],axis=0)
    chunks_col_final=np.append(col_chunk_main,[ylim2],axis=0)

#    nchild=int(nmaster/5)
    nchild=100
    main_block_iteration=0
    for iter_row in range(len(chunks_row_final[:]))[0:-1]:
       for iter_col in range(len(chunks_col_final[:]))[0:-1]:
           print("LOOP:", iter_row,iter_col)
           '''Write time series of the data for each master iteration '''
           in_file=inpath_final+'store_time_series_'+np.str(chunks_row_final[iter_row])+'_'+np.str(chunks_row_final[iter_row+1])+'_'+np.str(chunks_col_final[iter_col])+'_'+np.str(chunks_col_final[iter_col+1])+'_.nc'
           '''calculate trend from the each time series of master chunks''' 
           print('calculating trend for the chunk') 
           print(chunks_row_final[iter_row],chunks_row_final[iter_row+1],chunks_col_final[iter_col],chunks_col_final[iter_col+1],'***Row_SIZE***',chunks_row_final[iter_row+1]-chunks_row_final[iter_row],'***Col_SIZE***',chunks_col_final[iter_col+1]-chunks_col_final[iter_col]) 
           print(in_file)
           X_shape, Y_shape=chunks_row_final[iter_row+1]-chunks_row_final[iter_row],chunks_col_final[iter_col+1]-chunks_col_final[iter_col]
           
           '''Creation of sub chunks of 100 by 100 to estimate trends in Master 500 by 500 '''
           '''one can use 500 by 500 sub chunks to estimate trends if Master is by 1000 by 1000 '''
           mainchunks_x=np.arange(chunks_row_final[iter_row],chunks_row_final[iter_row+1],nchild)
           '''one can use 500 instead of 100 sub chunks to estimate trends if Master is by 1000 by 1000 '''
           mainchunks_y=np.arange(chunks_col_final[iter_col],chunks_col_final[iter_col+1],nchild)
           '''one can us e 500 instead of 100 sub chunks to estimate trends if Master is by 1000 by 1000 '''
    
           step_chunk_x=chunks_row_final[iter_row+1]-chunks_row_final[iter_row]
           step_chunk_y=chunks_col_final[iter_col+1]-chunks_col_final[iter_col]
           
           final_chunk_x=chunks_row_final[iter_row+1]
           final_chunk_y=chunks_col_final[iter_col+1]
           
           start_chunk_x=chunks_row_final[iter_row]
           start_chunk_y=chunks_col_final[iter_col]
          
           inputdata=[mainchunks_x,mainchunks_y]
           result_main_chunks = list(itertools.product(*inputdata))
           result_chunks=[(result_main_chunks[i][0],result_main_chunks[i][1],main_block_iteration,i, nchild) for i in range(len(result_main_chunks))]
           #print (result_chunks)
           iteration_final=len(result_chunks)-1
    
           ''' Applying the multiple processing here, with process of choice, i use 4 for local and 16 for lustre '''
           ''' Here we pass the arguments to the initializer for pool so we use in the function used in multiple processing, it can be changed differently '''
           nproc = 4
           with Pool(processes=nproc, initializer=init_worker_nc, initargs=(var_dict,X_shape,Y_shape,step_chunk_x,step_chunk_y,final_chunk_x,final_chunk_y,start_chunk_x,start_chunk_y,in_file,outpath_final,iteration_final,indx_seasons[0])) as pool:
               '''I am not returning results, usually you can return and write the results after multiprocessing '''
               print('pool started')
               results = pool.starmap(processInput_trends,[(result_chunks[i]) for i in range(len(result_chunks))])
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
    
    except Exception:
        traceback.print_exc()
        sys.exit()

