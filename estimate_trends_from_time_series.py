"""
Created on Fri Oct  4 09:47:45 2019

@author: moparthys

code to prepare tendencies based on Man-kendall test

"""


from datetime import datetime
import numpy as np
from scipy.stats import mstats
import mankendall_fortran_repeat_exp2 as m
from multiprocessing import Pool, Lock, current_process
import itertools
from netCDF4 import Dataset
from time import sleep
import sys, glob, os
import traceback
from timeit import default_timer as timer
import psutil


    
def init_worker_nc(var_dict, file_timeseries, file_outpath):
        '''
        Initialize the pool with boundaries of data to be read and write
        currently, the last version of the code uses only the input time series data
        Rest of arguments are not used
        '''
        
        # Using a dictionary to initialize all the boundaries of the grid
        var_dict['file_input_time_series'] = file_timeseries
        var_dict['file_output_tendencies'] = file_outpath
    
 
def processInput_trends(subchunk, parent_iteration, child_iteration):
    """This is the main file that calculate trends"""

    ## Debug tool to print process Ids
    process = psutil.Process(os.getpid())
    current = current_process()
    #print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')

    # Read the input time series file from main chunk, configured length of time X 500 X 500; it may vary if different chunks are used
    hdf_ts = Dataset(var_dict['file_input_time_series'], 'r', format='NETCDF4')

    # Create temporary storage with size of sub chunks in main chunk, currently configured 100 by 100 blocks
    var_temp_output = np.empty([*subchunk.dim,4])    
    var_temp_output[:] = np.nan
    # NaN matrix by default
    print ('### Chunk {} > subchunk {} started: COL: [{}:{}] ROW: [{}:{}]'.format(parent_iteration, child_iteration, *subchunk.get_limits('local', 'str')))

    
    b_deb = 0 # flag to print time profiling
    t00 = timer()
    t000 = timer()
    t_mean = 0.
    #print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')
    #for ii_sub in range(*subchunk.get_limits('local', 'tuple')[:2]):
    offsetx = subchunk.get_limits('local', 'tuple')[0]
    offsety = subchunk.get_limits('local', 'tuple')[2]

    for jj_sub in range(subchunk.dim[0]):
        # dimension of variable: time,x,y
        # preload all the y data here to avoid overhead due to calling Dataset.variables at each iteration in the inner loop
        data_test0 = hdf_ts.variables['time_series_chunk'][:,jj_sub+offsety,:]
        #data_test0 = hdf_ts.variables['time_series_chunk'][:500,sub_chunks_x[ii_sub],:]
        #print(f'ii_sub:{ii_sub} ({sub_chunks_x[0]},{sub_chunks_x[-1]})', f'len(sub_chunks_y):{len(sub_chunks_y)} ({sub_chunks_y[0]},{sub_chunks_y[-1]})')
        # debug
        #print(np.count_nonzero(np.isnan(data_test0)))

        for ii_sub in range(subchunk.dim[1]):
            if b_deb: print('---------------')
            if b_deb: print(f'{ii_sub} ({sub_chunks_x[0]},{sub_chunks_x[-1]})', f'{jj_sub} ({sub_chunks_y[0]},{sub_chunks_y[-1]})')
            
            t0 = timer()

            data_test = data_test0[:,ii_sub+offsetx]
    

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
            if 1:
                ## orinal mann-kendall test :
                p,z,Sn,nx = m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)
                # if data_test = [], the test return (p,z,Sn,nx) = (1.0, 0.0, 0.5, 0.0)
            else:
                ## other test
                p,z,Sn,nx = [0,0,0,0] 
                z = data_test.mean()

            if b_deb:
                print('t4', timer()-t0)
                t0 = timer()

            if b_deb: print('p,z,slope,nx', p,z,slope,nx)

            var_temp_output[jj_sub,ii_sub,0] = p
            var_temp_output[jj_sub,ii_sub,1] = z
            #var_temp_output[jj_sub,ii_sub,2] = slope
            var_temp_output[jj_sub,ii_sub,2] = Sn
            var_temp_output[jj_sub,ii_sub,3] = nx

        if 0:
            t_mean += timer()-t00
            #print(f't00 {ii_sub} {t_mean/(ii_sub+1)}')
            print(f't00.p{current._identity[0]}.it{ii_sub} {t_mean/(ii_sub+1):.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
            v = var_temp_output[ii_sub,:,:]
            for ii in range(4):
                print(f'{np.count_nonzero(np.isnan(v[:,ii]))/v[:,ii].size:.3f}', np.nanmin(v[:,ii]), np.nanmax(v[:,ii]))
            print(np.nanmin(v), np.nanmax(v))
            t00 = timer()
    
    #print(f't000tot.p{current._identity[0]} {timer()-t000:.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
    #print ('<<< Block completed: ROW: [{}:{}] COL: [{}:{}]'.format(tuple_limits1, tuple_limits1+step_chunk1, tuple_limits2, tuple_limits2+step_chunk2))
    write_string0 = (param.hash+"_CHUNK_" + np.str(parent_iteration)  
                     + "_SUBCHUNK_" + np.str(child_iteration)   
                     + "_" + '_'.join(subchunk.get_limits('global', 'str'))
                     + '.nc')
    write_string = var_dict['file_output_tendencies'] + '/' + write_string0
    write_string = os.path.normpath(write_string)
    # var_dict['file_output_tendencies'] prepared from the initializer routes the tendencies files to output destination as defined by product_tag previously

    if lock.acquire():
        #print ('Pool process Thread Locked:')
        #print ('Pool process Writing NetCDF:')
        nc_output = Dataset(write_string, 'w')
        xdim = nc_output.createDimension('X_dim', subchunk.dim[1])
        ydim = nc_output.createDimension('Y_dim', subchunk.dim[0])
        Vardim = nc_output.createDimension('Var', 4)
        
                     
        output_data = nc_output.createVariable('chunk_scores_p_val',  np.float64, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
        output_data = nc_output.createVariable('chunk_scores_z_val',  np.float64, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
        output_data = nc_output.createVariable('chunk_scores_Sn_val', np.float64, ('Y_dim','X_dim'), zlib=True, least_significant_digit=8)
        output_data = nc_output.createVariable('chunk_scores_length', np.float64, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
        
        nc_output.variables['chunk_scores_p_val'][:] = var_temp_output[:,:,0]
        nc_output.variables['chunk_scores_z_val'][:] = var_temp_output[:,:,1]
        nc_output.variables['chunk_scores_Sn_val'][:] = var_temp_output[:,:,2]
        nc_output.variables['chunk_scores_length'][:] = var_temp_output[:,:,3]
        
        nc_output.close() 
        lock.release()
        #print ('Pool process Thread Released:')
        #print ('Pool process worker finished: ')
   
        hdf_ts.close() 
    
        print ('Subchunk {} completed, save to {}'.format(child_iteration, write_string0))
        
        with open(os.path.normpath(var_dict['file_output_tendencies'] + '/filelist.txt'),'a') as fl:
            fl.write(write_string0+'\n')
        
        return None 



def main():

    input_path = param.input

    output_path = param.output
    
    product_tag = param.product_tag
    
    inpath_final = os.path.normpath(input_path+os.sep + product_tag + os.sep) + os.sep
    outpath_final = os.path.normpath(output_path + os.sep + product_tag + os.sep) + os.sep

    # Empty list of data files used further for plotting
    with open(outpath_final + '/filelist.txt', 'w'): pass

    
    nchild = 100
    main_block_iteration = 0
    for chunk in param.chunks.list:
        #print("Chunk:", iter_row,iter_col)
        # Calculate trend from the each time series of master chunks 
        in_file = inpath_final+param.hash+'_timeseries_'+ '_'.join(chunk.get_limits('global','str'))+'.nc'
        print('> calculating trend for the chunk:') 
        print(in_file)
        print('***Row/y_SIZE***', chunk.dim[0], '***Col/x_SIZE***', chunk.dim[1]) 
        
         
        chunk.subdivide(nchild)
        #for sub in chunk.list:
        #    print(sub.dim)

        result_chunks = [(sub, main_block_iteration, i) for i,sub in enumerate(chunk.list)]
    
        # Applying the multiple processing here, with process of choice, i use 4 for local and 16 for lustre 
        # Here we pass the arguments to the initializer for pool so we use in the function used in multiple processing, it can be changed differently 
        with Pool(processes=param.nproc, initializer=init_worker_nc, initargs=(var_dict, in_file, outpath_final)) as pool:
            # I am not returning results, usually you can return and write the results after multiprocessing
            print('pool started')
            results = pool.starmap(processInput_trends, [(i) for i in result_chunks])
            pool.close()
            pool.join()
            print('pool cleaned')

        # increase the main block iteration
        main_block_iteration += 1


def compute_trends(*largs):
    # Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main

    global lock 
    global var_dict 

    lock = Lock()
    var_dict = {}

    class ArgsTmp:
        def __init__(self, h, inp, outp, prod, chunks, np):
            self.hash = h
            self.input = inp
            self.output = outp
            self.product_tag = prod
            self.nproc = np
            self.chunks = chunks



    global param 
    param = ArgsTmp(*largs)
    main()



if __name__ == "__main__":
    try:
        # Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main
        lock = Lock()
        var_dict = {}
        param = parse_args()
        main()
    
    except Exception:
        traceback.print_exc()
        sys.exit()

