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

if 0:
    from functools import wraps
    import errno
    import os
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
        def decorator(func):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)
    
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            return wraps(func)(wrapper)
        return decorator
    
    @timeout(2.0)
    def mk_test_timeout(data_test):
        return m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)


    
 
def processInput_trends(subchunk, parent_iteration, child_iteration):
    """This is the main file that calculate trends"""

    #print('INFO: see pid.<pid>.out to monitor trend computation progress')
    #sys.stdout = open('pid.'+str(os.getpid()) + '.out', 'w')
    
    #print('INFO: see trend.out to monitor trend computation progress')
    #sys.stdout = open('trend.out', 'a')

    ## Debug tool to print process Ids
    process = psutil.Process(os.getpid())
    current = current_process()
    print(process, current._identity, '{} Mo'.format(process.memory_info().rss/1024/1024))

    if subchunk.input=='box':
        print('### Chunk {} > subchunk {} started: COL: [{}:{}] ROW: [{}:{}]'.format(parent_iteration, child_iteration, *subchunk.get_limits('local', 'str')))
        write_string0 = (param.hash+"_CHUNK" + np.str(parent_iteration)  
                         + "_SUBCHUNK" + np.str(child_iteration)   
                         + "_" + '_'.join(subchunk.get_limits('global', 'str'))
                         + '.nc')
        subchunk_fname = param.output_path / write_string0
        ## Check if cache file already exists and must be overwritten
        if not param.b_delete:
            if subchunk_fname.is_file():
                print ('INFO: {} already exists. Use -d option to overwrite it.'.format(write_string0))
                return

    elif subchunk.input=='points':
        print('### Chunk {} > subchunk {} started.'.format(parent_iteration, child_iteration))
        print(param.input_file)
        str_date_range = param.input_file.stem.replace('timeseries','')
        write_string0 = 'merged_trends{}.nc' .format(str_date_range) 
        subchunk_fname = param.output_path / write_string0
        ## Result file is always overwritten in the case of point input

    ## Read the input time series file from main chunk, configured length of time X 500 X 500; it may vary if different chunks are used
    hdf_ts = Dataset(param.input_file, 'r', format='NETCDF4')

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

    tsvar = 'time_series_chunk'
    tsvar = 'AL_DH_BB'

    for jj_sub in range(subchunk.dim[0]):
    #for jj_sub in range(61,80): #debug
        # dimension of variable: time,x,y
        # preload all the y data here to avoid overhead due to calling Dataset.variables at each iteration in the inner loop
        data_test0 = hdf_ts.variables[tsvar][:,jj_sub+offsety,offsetx:offsetx+subchunk.dim[1]]
        #data_test0 = hdf_ts.variables[tsvar][:500,sub_chunks_x[ii_sub],:]

        for ii_sub in range(subchunk.dim[1]):
        #for ii_sub in range(55,100):
            if b_deb: print('---------------')
            if b_deb: print('jj: {} - ii: {} '.format(jj_sub, ii_sub))
            
            t0 = timer()

            data_test = data_test0[:,ii_sub]

            ## remove tie group
            data_test[1:][np.diff(data_test)==0.] = np.nan
    
            #data_test=hdf_ts.variables[tsvar][:,sub_chunks_x[ii_sub],sub_chunks_y[jj_sub]]
            slope=999.0

            if b_deb:
                print('t0', timer()-t0)
                t0 = timer()

            if b_deb: print('Data valid:', data_test.size - np.isnan(data_test).sum(), '/', data_test.size)
            
            if 0:
                print('Use mstats')
                data_sen=np.ma.masked_array(data_test, mask=np.isnan(data_test))
                t0 = timer()
                slope, intercept, lo_slope, up_slope = mstats.theilslopes(data_sen, alpha=0.1)
                print('slope, intercept, lo_slope, up_slop:')
                print(slope, intercept, lo_slope, up_slope)
                if b_deb: print('t02', timer()-t0)
                np.savetxt('data_test.dat', data_test.T)
                sys.exit()
                t0 = timer()

            # this mstats give correct slope and is consistent with python man-kendall score of Sn; this is fast than Fortran'''
            # stats.theilslopes is giving incorrect values when NaN are inside data'''
            
            if b_deb:
                print('t2', timer()-t0)
                t0 = timer()

            if 1:
                ## orinal mann-kendall test :
                bla = data_test[~np.isnan(data_test)]
                if bla.size > 0:
                    #print('min/mean/max/nb/nb_unique', bla.min(), bla.mean(), bla.max(), len(bla), len(np.unique(bla)))
                    #print(bla)
                    if len(np.unique(bla))==1:
                        p,z,Sn,nx = [0,0,0,0] 
                    else:
                        #data_test = data_test[-10:] # debug line to speed up
                        
                        #try:
                        #    p,z,Sn,nx = mk_test_timeout(data_test)
                        #except TimeoutError as e:
                        #    print('timeout!')
                        #    p,z,Sn,nx = [0,0,0,0] 
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
                    
                print('t4=', t4)
                if 0:
                    import matplotlib.pyplot as plt
                    plt.clf()
                    plt.plot(bla)
                    plt.ylim(0,6.1)
                    ti1 = '{}/{} - {:.3f} s'.format(jj_sub, ii_sub, t4)
                    ti2 = 'min/mean/max/nb/nb_unique {:.3f} {:.3f} {:.3f} {} {}'.format(bla.min(), bla.mean(), bla.max(), len(bla), len(np.unique(bla)))
                    ti3 = 'slope: {}'.format(Sn)
                    plt.title(ti1+'\n'+ti2+'\n'+ti3)
                    if Sn==0.0:
                        plt.savefig('bla.Sn0.{}.{}.png'.format(jj_sub, ii_sub))
                    else:
                        plt.savefig('bla.{}.{}.png'.format(jj_sub, ii_sub))
                t0 = timer()

            if b_deb: print('p,z,slope,nx', p,z,slope,nx)
            if b_deb: print('p,z,Sn,nx', p,z,Sn,nx)

            var_temp_output[jj_sub,ii_sub,0] = p
            var_temp_output[jj_sub,ii_sub,1] = z
            #var_temp_output[jj_sub,ii_sub,2] = slope
            var_temp_output[jj_sub,ii_sub,2] = Sn
            var_temp_output[jj_sub,ii_sub,3] = nx

       
        ## Print efficiency stats
        if (jj_sub+1)%print_freq==0:
            elapsed = timer()-t00
            data_stat = hdf_ts.variables[tsvar][:,jj_sub+1+offsety-print_freq:jj_sub+1+offsety,offsetx:offsetx+subchunk.dim[1]]
            valid = 100.*(data_stat.size - np.count_nonzero(np.isnan(data_stat)))/data_stat.size
            eff = 1e6*elapsed/data_stat.size
            #print(subchunk.dim, data_test0.shape)
            print('{} : {}.{}.block[{}-{}] : {:.3f}s elapsed : {:.3f} us/pix/date : {:.2f}% valid'.format(datetime.datetime.now(), parent_iteration, child_iteration, jj_sub+1-print_freq, jj_sub+1, elapsed, eff, valid))

            t00 = timer()
            sys.stdout.flush()

        if 0:
            t_mean += timer()-t00
            #print(f't00 {ii_sub} {t_mean/(ii_sub+1)}')
            #print(f't00.p{current._identity[0]}.it{ii_sub} {t_mean/(ii_sub+1):.3f}s {process.memory_info().rss/1024/1024:.2f}Mo')
            print('t00.p{}.it{} {:.3f}s {:.2f}Mo'.format(current._identity[0], ii_sub, t_mean/(ii_sub+1), process.memory_info().rss/1024/1024))
            v = var_temp_output[ii_sub,:,:]
            for ii in range(4):
                #print(f'{np.count_nonzero(np.isnan(v[:,ii]))/v[:,ii].size:.3f}', np.nanmin(v[:,ii]), np.nanmax(v[:,ii]))
                print('{:.3f}'.format(np.count_nonzero(np.isnan(v[:,ii]))/v[:,ii].size), np.nanmin(v[:,ii]), np.nanmax(v[:,ii]))
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

    print('t000tot.p{} {:.3f}s {:.2f}Mo'.format(current._identity, timer()-t000, process.memory_info().rss/1024/1024))


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
    
        print ('Subchunk {} completed, save to {}'.format(child_iteration, subchunk_fname))
        
        return None 



def main():

    # Calculate trend from the each time series of master chunks 
    
    nchild = 100
    main_iteration = 0
    
    for chunk in param.chunks.list:
       
        ## Check if it's a merged file or an original one
        if param.product.endswith('_MERGED'):
            list_of_paths = param.input_path.glob('*')
            latest_path = max(list_of_paths, key=lambda p: p.stat().st_ctime)
            param.input_file = latest_path
        else:
            param.input_file = param.input_path / (param.hash+'_timeseries_'+ '_'.join(chunk.get_limits('global','str'))+'.nc')
        print('> calculating trend for the chunk:') 
        print(param.input_file.as_posix())

        if chunk.input=='box':
            print('***Row/y_SIZE***', chunk.dim[0], '***Col/x_SIZE***', chunk.dim[1]) 
            
            # Subdivide each chunk into subchunks, the latter being then possibly multiprocessed
            chunk.subdivide(nchild)
        
        elif chunk.input=='points':
            print('***', 'SIZE {} points'.format(chunk.dim[1]))

        result_chunks = [(sub_chunk, main_iteration, sub_iteration) for sub_iteration,sub_chunk in enumerate(chunk.list)]
        
        #### Use multiprocessing
        if 0:

            # Applying the multiple processing here, with process of choice, i use 4 for local and 16 for lustre 
            with Pool(processes=param.nproc) as pool:
                print('pool started')
                results = pool.starmap(processInput_trends, result_chunks)
                pool.close()
                pool.join()
                print('pool cleaned')

        #### or do it sequentially
        else:

            for i in result_chunks:
                processInput_trends(*i)


        # increase the main iteration
        main_iteration += 1


def compute_trends(*args):
    # Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main

    global lock 

    lock = Lock()

    class ArgsTmp:
        def __init__(self, prod, chunks, np, b_delete, config):
            self.hash = prod.hash
            self.input = pathlib.Path(config['output_path']['extract'])
            self.output = pathlib.Path(config['output_path']['trend'])
            self.product = prod.name
            self.nproc = np
            self.chunks = chunks
            self.b_delete = b_delete
        
            self.input_path = self.input / prod.name 
            self.input_file = '' 
            
            self.output_path = self.output / prod.name
            # Make dir if not exists
            self.output_path.mkdir(parents=True, exist_ok=True)

    # Erase output log file
    fic = open('trend.out','w')
    fic.close()

    global param 
    param = ArgsTmp(*args)
    main()



if __name__ == "__main__":
    try:
        # Lock is the module from multiprocessing to allow the write of netcdf files one at a time to avoid conflict, first invocation here in main
        lock = Lock()
        param = parse_args()
        main()
    
    except Exception:
        traceback.print_exc()
        sys.exit()

