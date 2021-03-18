"""
Created on Fri Oct  4 09:47:45 2019

@author: moparthys

code to prepare tendencies based on Man-kendall test

"""


from datetime import datetime
import numpy as np
import pandas as pd
import mankendall_fortran_repeat_exp2 as m
from multiprocessing import Pool, Lock, current_process
import itertools
import h5py
from time import sleep
import sys, glob, os
import traceback
from timeit import default_timer as timer
import datetime
import psutil
import pathlib

    
 
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
        write_string0 = 'merged_trends{}.h5'.format(str_date_range) 
        subchunk_fname = param.output_path / write_string0
        ## Result file is always overwritten in the case of point input

    ## Read the input time series file from main chunk, configured length of time X 500 X 500; it may vary if different chunks are used
    hdf_ts = h5py.File(param.input_file, 'r')

    pt_names = hdf_ts['meta/point_names'][:]
    globid = hdf_ts['meta/global_id'][:]

    ## Create temporary storage with size of sub chunks in main chunk, currently configured 100 by 100 blocks
    var_temp_output = np.empty([*subchunk.dim,4])    
    var_temp_output[:] = np.nan # NaN matrix by default

    ## Parameters for the loop
    b_deb = 1 # flag to print time profiling
    t00 = timer()
    t000 = timer()
    t_mean = 0.
    #print(current._identity, f'{process.memory_info().rss/1024/1024} Mo')
    offsetx = subchunk.get_limits('local', 'tuple')[0]
    offsety = subchunk.get_limits('local', 'tuple')[2]

    print_freq = 20

    tab_prof_valid = []
    tab_prof_zero = []

    hf = h5py.File(subchunk_fname, 'w')

    for tsvar in hdf_ts['vars'].keys():
        print("\n---", tsvar)

        for jj_sub in range(subchunk.dim[0]):
        #for jj_sub in range(61,80): #debug
            # dimension of variable: time,x,y
            # preload all the y data here to avoid overhead due to calling Dataset.variables at each iteration in the inner loop
            data_test0 = hdf_ts['vars/'+tsvar][:,jj_sub+offsety,offsetx:offsetx+subchunk.dim[1]]
            #data_test0 = hdf_ts.variables[tsvar][:500,sub_chunks_x[ii_sub],:]

            if 1:
                #z,Sn,nx = pandas_wrapper(data_test0, pt_names, globid,  b_deb)
                res = pandas_wrapper(data_test0, pt_names, globid, 0)

            else:
                #p,z,Sn,nx = legacy_wrapper(data_test0, subchunk, b_deb)
                res = legacy_wrapper(data_test0, subchunk, b_deb)


            var_temp_output[jj_sub] = res

            #var_temp_output[jj_sub,:,0] = p
            #var_temp_output[jj_sub,:,1] = z
            #var_temp_output[jj_sub,:,2] = Sn
            #var_temp_output[jj_sub,:,3] = nx

           
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


        
        hf.create_dataset(tsvar+'/pval', data=var_temp_output[:,:,0])
        hf.create_dataset(tsvar+'/zval', data=var_temp_output[:,:,1])
        hf.create_dataset(tsvar+'/slope', data=var_temp_output[:,:,2])
        hf.create_dataset(tsvar+'/len', data=var_temp_output[:,:,3])

    hf.close() 
    
    print ('Subchunk {} completed, save to {}'.format(child_iteration, subchunk_fname))
    
    return None 


def pandas_wrapper(data_test0, pt_names, globid, b_deb):
    """
    Pandas wrapper that vectorize stat processing
    """

    from tools import SimpleTimer
    ti = SimpleTimer()  

    b_deb = 0
    b_plot = 0

    import matplotlib.pyplot as plt
    def save_heatmap(A, suffix):
        print("Plot heat map...")
        plt.clf()
        plt.imshow(A)
        plt.savefig("heatmap_{}.png".format(suffix))

    ## Option to print more data from big DataFrame
    #pd.set_option('display.max_rows', None)  
    #pd.set_option('display.max_columns', None)  
    #pd.set_option('display.expand_frame_repr', False)
    #pd.set_option('max_colwidth', -1)
    
    if b_deb: print(pt_names)

    df = pd.DataFrame(data_test0, columns=pt_names, index=[globid//36, globid%36])
    if b_plot: save_heatmap(df.values.T, "before")
    df2 = df.unstack()
    if b_deb: print(df2)
    
    ### Check that for each dekad, there is at least 70% of non-NaN observations. 
    
    ti()
    #valid_dekad = df2.apply(lambda x: (np.count_nonzero(~np.isnan(x))/df2.shape[0])>0.7)
    valid_dekad = ((~np.isnan(df2.values)).sum(axis=0)/df2.shape[0])>0.7
    ti('t01:apply_0.7filter')        
    #print("valid_dekad=", valid_dekad)
    # Number of valid dekad:
    if b_deb:
        print("valid_dekad.shape=", valid_dekad.shape)
        dekad_len = df2.shape[0]
        n_dekad = df2.shape[1]
        valid = np.count_nonzero(df2.apply(lambda x: (np.count_nonzero(~np.isnan(x))/dekad_len)>0.7).values.ravel())
        print("valid,n_dekad=", valid,n_dekad )
        print("% of dekad discarded:", 100*(1-valid/n_dekad)) 


    ### Set the dekad full of NaN if the 70% threshold is not valid
   
    ti()
    idv = np.arange(len(valid_dekad))[~valid_dekad] # Get indices of valid columns
    npdf2 = df2.values
    npdf2[:,idv] = np.nan # using numpy is really faster than doing the same in pandas
    df2 = pd.DataFrame(npdf2, columns=df2.columns, index=df2.index) # copy the original df2

    ti('t2:set_nan')        
    
    ### Commpute z-score
    
    df_zman = (df2-df2.mean())/df2.std() # default ddof=1 or df2.std(ddof=0) does not change final p-value
    if b_deb: print(df_zman)
    
    ### Back to continuous time series, keep all NaN and remove only leading and trailing one.
    
    df_res = df_zman.stack(dropna=False)
    df_res = df_res.loc[df_res.first_valid_index():df_res.last_valid_index()]
    if b_deb: print(df_res)
    if b_plot: save_heatmap(df_res.values.T, "after")

    ### Test and apply MKtest on zscaled data (we just want p-value here)
    
    def preproc(y):
        # Second 70% threshold: compute MK test only if 70% of valid data
        if (np.count_nonzero(np.isnan(y))/len(y))>0.3:
            if b_deb: print("WARNING: More than 30%  of NaN")
            return tuple([np.nan]*4)
        return m.mk_trend(len(y), np.arange(len(y)), y, 1)
    
    ti()
    df_mk = df_res.apply(preproc)
    ti('t32:apply_mk1')        
    #print(df_mk) # df_mk is a Series, not a DataFrame 
    df_mk = pd.DataFrame.from_items(zip(df_mk.index, df_mk.values)).T
    df_mk.columns = ['p','z','sn','nx']
    #print(df_mk.sort_values(by=['p']))
    if b_deb: print(df_mk)

    ### Finally compute Slope only for data with p-value < 0.05

    def proc(y):
        #print(p_test[y.name])
        if not p_test[y.name]:
            if b_deb: print("INFO: p-value > 0.05 for {}".format(y.name))
            return tuple([np.nan]*4)
        return m.mk_trend(len(y), np.arange(len(y)), y, 2)
    
    p_test = df_mk['p'] < 0.05
    if b_deb: print(p_test)
    ti()
    df_phy = df.apply(proc)
    ti('t42:apply_mk2')        
    df_phy = pd.DataFrame.from_items(zip(df_phy.index, df_phy.values)).T
    df_phy.columns = ['p','z','sn','nx']
    if b_deb: print(df_phy)

    ### return [p_z, z_z, sn_phy, nx_z]
    df_mk['sn'] = df_phy['sn']

    ti.show()

    return df_mk.values


def legacy_wrapper(data_test0, subchunk, b_deb):
    """
    Legacy fortran wrapper that loop on each pixel
    """

    res = np.empty([subchunk.dim[1],4])    
    res[:] = np.nan # NaN matrix by default

    for ii_sub in range(subchunk.dim[1]):
        #if b_deb: print('---------------')
        #if b_deb: print('jj: {} - ii: {} '.format(jj_sub, ii_sub))
        
        data_test = data_test0[:,ii_sub]

        ## remove tie group
        data_test[1:][np.diff(data_test)==0.] = np.nan
    
        if b_deb: print('Data valid:', data_test.size - np.isnan(data_test).sum(), '/', data_test.size)
        
        if 1:
            ## original mann-kendall test :
            bla = data_test[~np.isnan(data_test)]
            if bla.size > 0:
                #print('min/mean/max/nb/nb_unique', bla.min(), bla.mean(), bla.max(), len(bla), len(np.unique(bla)))
                #print(bla)
                if len(np.unique(bla))==1:
                    p,z,Sn,nx = [0,0,0,0] 
                else:

                    p,z,Sn,nx = m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)
            else:
                p,z,Sn,nx = m.mk_trend(len(data_test), np.arange(len(data_test)), data_test)
                # if data_test = [], the test return (p,z,Sn,nx) = (1.0, 0.0, 0.5, 0.0)
        else:
            ## other test
            p,z,Sn,nx = [0,0,0,0] 
            z = data_test.mean()

        #if b_deb: print('p,z,Sn,nx', p,z,Sn,nx)
        
        res[ii_sub,0] = p
        res[ii_sub,1] = z
        res[ii_sub,2] = Sn
        res[ii_sub,3] = nx
    
    if b_deb:
        print('p,z,Sn,nx')
        print(res)

    #return (res[:,0], res[:,1], res[:,2], res[:,3])
    return res



def main(*args):

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




if __name__ == "__main__":
    try:
        main()
    
    except Exception:
        traceback.print_exc()
        sys.exit()

