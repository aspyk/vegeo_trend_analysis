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
import datetime
import psutil
import pathlib


    
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

    #print('INFO: see pid.<pid>.out to monitor trend computation progress')
    #sys.stdout = open('pid'+str(os.getpid()) + '.out', 'w')
    #sys.stdout = open('pid.out', 'a')

    print ('### Chunk {} > subchunk {} started: COL: [{}:{}] ROW: [{}:{}]'.format(parent_iteration, child_iteration, *subchunk.get_limits('local', 'str')))

    ## Debug tool to print process Ids
    process = psutil.Process(os.getpid())
    current = current_process()
    #print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')

    write_string0 = (param.hash+"_CHUNK_" + np.str(parent_iteration)  
                     + "_SUBCHUNK_" + np.str(child_iteration)   
                     + "_" + '_'.join(subchunk.get_limits('global', 'str'))
                     + '.nc')
    subchunk_fname = var_dict['file_output_tendencies'] + '/' + write_string0
    subchunk_fname = os.path.normpath(subchunk_fname)

    ## Check if cache file already exists and must be overwritten
    if not param.b_delete:
        if pathlib.Path(subchunk_fname).is_file():
            print ('Subchunk {} already exists in {}. Use -d option to overwrite it.'.format(child_iteration, write_string0))
            return

    ## Read the input time series file from main chunk, configured length of time X 500 X 500; it may vary if different chunks are used
    hdf_ts = Dataset(var_dict['file_input_time_series'], 'r', format='NETCDF4')

    ## Create temporary storage with size of sub chunks in main chunk, currently configured 100 by 100 blocks
    var_temp_output = np.empty([*subchunk.dim,4])    
    var_temp_output[:] = np.nan
    # NaN matrix by default

    
    ## Parameters for te loop
    b_deb = 0 # flag to print time profiling
    t00 = timer()
    t000 = timer()
    t_mean = 0.
    #print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')
    offsetx = subchunk.get_limits('local', 'tuple')[0]
    offsety = subchunk.get_limits('local', 'tuple')[2]

    print_freq = 20

    tab_prof_valid = []
    tab_prof_zero = []

    for jj_sub in range(subchunk.dim[0]):
    #for jj_sub in range(28,30): #debug
        # dimension of variable: time,x,y
        # preload all the y data here to avoid overhead due to calling Dataset.variables at each iteration in the inner loop
        data_test0 = hdf_ts.variables['time_series_chunk'][:,jj_sub+offsety,offsetx:offsetx+subchunk.dim[1]]
        #data_test0 = hdf_ts.variables['time_series_chunk'][:500,sub_chunks_x[ii_sub],:]

        for ii_sub in range(subchunk.dim[1]):
            if b_deb: print('---------------')
            if b_deb: print('jj: {} - ii: {} '.format(jj_sub, ii_sub))
            
            t0 = timer()

            data_test = data_test0[:,ii_sub]
    

            #data_test=hdf_ts.variables['time_series_chunk'][:,sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]
            slope=999.0

            if b_deb:
                print('t0', timer()-t0)
                t0 = timer()

            if b_deb: print('Data:', f'{data_test.size - np.isnan(data_test).sum()}/{data_test.size}')
            
            if 0:
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
                bla = data_test[data_test!=999]
                if bla.size > 0:
                    #print('min/mean/max', bla.min(), bla.mean(), bla.max(), len(bla))
                    if len(np.unique(bla))==1:
                        p,z,Sn,nx = [0,0,0,0] 
                    else:
                        #data_test = data_test[-10:] # debug line to speed up
                        p,z,Sn,nx = m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)
                else:
                    p,z,Sn,nx = m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)
                    # if data_test = [], the test return (p,z,Sn,nx) = (1.0, 0.0, 0.5, 0.0)
            else:
                ## other test
                p,z,Sn,nx = [0,0,0,0] 
                z = data_test.mean()

            if b_deb:
                t4 = timer()-t0
                if bla.size>0:
                    if bla.mean()==0.0:
                        tab_prof_zero.append(t4)
                    else:
                        tab_prof_valid.append(t4)
                    
                print(t4)
                t0 = timer()

            if b_deb: print('p,z,slope,nx', p,z,slope,nx)

            var_temp_output[jj_sub,ii_sub,0] = p
            var_temp_output[jj_sub,ii_sub,1] = z
            #var_temp_output[jj_sub,ii_sub,2] = slope
            var_temp_output[jj_sub,ii_sub,2] = Sn
            var_temp_output[jj_sub,ii_sub,3] = nx

       
        ## Print efficiency stats
        if (jj_sub+1)%print_freq==0:
            elapsed = timer()-t00
            data_stat = hdf_ts.variables['time_series_chunk'][:,jj_sub+1+offsety-print_freq:jj_sub+1+offsety,offsetx:offsetx+subchunk.dim[1]]
            valid = 100.*(data_stat.size - np.count_nonzero(np.isnan(data_stat)))/data_stat.size
            eff = 1e6*elapsed/data_stat.size
            #print(subchunk.dim, data_test0.shape)
            print('{} : {}.{}.block[{}-{}] : {:.3f}s elapsed : {:.3f} us/pix/date : {:.2f}% valid'.format(datetime.datetime.now(), parent_iteration, child_iteration, jj_sub+1-print_freq, jj_sub+1, elapsed, eff, valid))

            t00 = timer()

        if 0:
            t_mean += timer()-t00
            #print(f't00 {ii_sub} {t_mean/(ii_sub+1)}')
            print(f't00.p{current._identity[0]}.it{ii_sub} {t_mean/(ii_sub+1):.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
            v = var_temp_output[ii_sub,:,:]
            for ii in range(4):
                print(f'{np.count_nonzero(np.isnan(v[:,ii]))/v[:,ii].size:.3f}', np.nanmin(v[:,ii]), np.nanmax(v[:,ii]))
            print(np.nanmin(v), np.nanmax(v))
            t00 = timer()
   
    if b_deb:
    #if 1:
        valid = np.array(tab_prof_valid)
        zero = np.array(tab_prof_zero)

        #print('valid:', valid.size, valid.min(), valid.mean(), valid.max())
        #print('zero:', zero.size, zero.min(), zero.mean(), zero.max())
        print(valid.mean())
        print(zero.mean())
        #return
        #sys.exit()

    print(f't000tot.p{current._identity[0]} {timer()-t000:.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')

    if lock.acquire():
        #print ('Pool process Thread Locked:')
        #print ('Pool process Writing NetCDF:')
        nc_output = Dataset(subchunk_fname, 'w')
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
        
        return None 



def main():

    input_path = param.input
    output_path = param.output
    product_tag = param.product_tag
    
    inpath_final = os.path.normpath(input_path+os.sep + product_tag + os.sep) + os.sep
    outpath_final = os.path.normpath(output_path + os.sep + product_tag + os.sep) + os.sep

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

        result_chunks = [(sub, main_block_iteration, i) for i,sub in enumerate(chunk.list)]
        
        # Use multiprocessing
        if 1:

            # Applying the multiple processing here, with process of choice, i use 4 for local and 16 for lustre 
            # Here we pass the arguments to the initializer for pool so we use in the function used in multiple processing, it can be changed differently 
            with Pool(processes=param.nproc, initializer=init_worker_nc, initargs=(var_dict, in_file, outpath_final)) as pool:
                # I am not returning results, usually you can return and write the results after multiprocessing
                print('pool started')
                results = pool.starmap(processInput_trends, [(i) for i in result_chunks])
                pool.close()
                pool.join()
                print('pool cleaned')

        ## or so it sequentially
        else:

            init_worker_nc(var_dict, in_file, outpath_final)
            for i in result_chunks:
                processInput_trends(*i)


        # increase the main block iteration
        main_block_iteration += 1


def compute_trends(*args):
    # Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main

    global lock 
    global var_dict 

    lock = Lock()
    var_dict = {}

    class ArgsTmp:
        def __init__(self, h, inp, outp, prod, chunks, np, b_delete):
            self.hash = h
            self.input = inp
            self.output = outp
            self.product_tag = prod
            self.nproc = np
            self.chunks = chunks
            self.b_delete = b_delete



    global param 
    param = ArgsTmp(*args)
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

