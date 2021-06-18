import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.ticker as ticker
import h5py
import pandas as pd
from scipy import interpolate
from scipy import optimize
import datetime
from itertools import cycle
import pathlib
import mankendall_fortran_repeat_exp2 as m

from timeit import default_timer as timer
import os,sys

import random
import pprint

#pd.set_option('display.max_rows', 500)

class SimpleTimer():
    def __init__(self):
        self.t0 = timer()
        self.res = []

    def __call__(self, msg=''):
        dt = timer()-self.t0
        self.res.append([msg, dt])
        print(msg, dt)
        self.t0 = timer()

    def show(self):
        print("** Timing summary **")
        for r in self.res:
            print("{0} {1}".format(*r))

def snht(xin, return_array=False):
    ## Remove NaNs 
    nan_mask = ~np.isnan(xin)
    idx = np.arange(len(xin))[nan_mask]
    x = xin[nan_mask]

    n = len(x)
    k = np.arange(1, n)
    s = x.cumsum()[:-1]
    rs = x[::-1].cumsum()[::-1][1:]

    mean = x.mean()
    std = x.std(ddof=1)

    z1 = ((s - k * mean) / std) / k
    z2 = ((rs - k[::-1] * mean) / std) / (n - k)
    T = k * z1 * z1 + (n - k) * z2 * z2 

    nvalid = n
    nnan = len(xin)-n

    tloc = T.argmax()+1

    mu1 = x[:tloc].mean()
    mu2 = x[tloc:].mean()

    if return_array:
        # Duplicate first value of T to fit the shape of valid data
        Tarr = np.full(xin.shape, np.nan)
        Tarr[nan_mask] = np.concatenate([T[:1],T])
        return Tarr, nvalid, nnan
    else:
        return T.max(), idx[tloc], nvalid, nnan, mu1, mu2, ((mu2-mu1)/mu1)*100

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

def mc_p_value(n, sim): 
    '''
    Monte carlo simulation for p-value calculation
    '''
    rand_data = np.random.normal(0, 1, [sim, n])
    # print(rand_data.shape)
    res = np.asarray(list(map(snht, rand_data, [False]*sim)))
    # p_val = (res[:,0] > stat).sum() / sim
    
    return res

def compute_pval_from_cache(stats, nval):
    with h5py.File('mc_cache_20k.h5','r') as hf:
        nmin = hf['tvalue'].attrs['nmin']
        nmax = hf['tvalue'].attrs['nmax']
        nsim = hf['tvalue'].attrs['nsim']
        cache = hf['tvalue'][:]

    print("cache.shape=", cache.shape)

    p_val = np.array([(cache[n-nmin] > s).sum() / nsim for s,n in zip(stats, nval.astype('int'))])
    return p_val

def compute_pval_from_cache_scalar(stats, nval):
    with h5py.File('mc_cache_20k.h5','r') as hf:
        nmin = hf['tvalue'].attrs['nmin']
        nmax = hf['tvalue'].attrs['nmax']
        nsim = hf['tvalue'].attrs['nsim']
        cache = hf['tvalue'][nval-nmin]

    p_val = (cache > stats).sum() / nsim 
    return p_val

def create_mc_cache(nmin=10, nmax=1600, sim=20000):
    """Compute normal reference for p value"""

    mc_cache = np.empty((nmax-nmin, sim))
    res_root = []
    for n in range(nmin, nmax):

        t0 = datetime.datetime.now()
        res = mc_p_value(n, sim)[:,0]
        print(n, datetime.datetime.now()-t0)
        mc_cache[n-nmin] = res

    print('--- Save h5 file')
    with h5py.File('tmp_mc_cache_{:d}.h5'.format(sim),'w') as hf:
        hf['tvalue'] = mc_cache
        hf['tvalue'].attrs['nmin'] = nmin
        hf['tvalue'].attrs['nmax'] = nmax
        hf['tvalue'].attrs['nsim'] = sim

def test_fake_data():
    x = []
    x.append(5.1*np.random.rand(250)+ 0.0)
    x.append(5.1*np.random.rand(250) + 0.5)
    # x.append(0.1*np.random.rand(300) - 1.)
    # x.append(0.1*np.random.rand(250) + 0.0 )
    x = np.concatenate(x)

    ti = SimpleTimer()

    #[snht(x) for i in range(1000)]

    ti('end')

    #t, tmax, tmaxloc , nvalid, nnan= snht(x, return_array=True)
    #print(tmax, tmaxloc)

    if 1:
        ## Compute normal reference for p value 
        nmin = 10
        nmax= 1600
        sim = 20000

        mc_cache = np.empty((nmax-nmin, sim))
        res_root = []
        for n in range(nmin, nmax):

            t0 = datetime.datetime.now()
            res = mc_p_value(n, sim)[:,0]
            print(n, datetime.datetime.now()-t0)
            mc_cache[n-nmin] = res
            continue

            # print('--- Compute normal stat repartition')
            test_val = np.linspace(0,res.max(),50)
            test_p = np.array([(res > i).sum()/sim for i in test_val])
            
            f = interpolate.interp1d(test_val, test_p, kind='quadratic')
            def f2(x):
                return f(x)-0.05
            sol = optimize.root_scalar(f2, bracket=[0, test_val.max()], method='brentq')

            # print(sol.root)
            res_root.append([n, sol.root])
            print(res_root[-1])

        if 1:
            print('--- Save h5 file')
            with h5py.File('tmp_mc_cache_20k.h5','w') as hf:
                hf['tvalue'] = mc_cache
                hf['tvalue'].attrs['nmin'] = nmin
                hf['tvalue'].attrs['nmax'] = nmax
                hf['tvalue'].attrs['nsim'] = sim

            sys.exit()

        res_root = np.array(res_root)

        print('--- Save h5 file')
        with h5py.File('mc_pval_nlim_for_5pct.h5','w') as hf:
            hf['pval_nvalid'] = res_root.T[0]
            hf['pval_lim'] = res_root.T[1]

    ## Plot T limit value for p value = 0.05
    if 1:
        print('--- Plot')
        fig, ax1 = plt.subplots()
        ax1.plot(res_root.T[0], res_root.T[1])
        plt.savefig('output_snht_pvalue.png')

    ## Plot MC normal p value
    if 0:
        print('--- Plot')
        fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        x2 = np.linspace(0, res.max(), 200)
        y2 = list(map(f2, x2))
        ax1.plot(test_val, test_p-0.05)
        ax1.plot(x2, y2)
        ax1.axhline(0.0, c='r')
        plt.show()

    ## Plot T curve
    if 0:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(x)
        ax2.plot(t, c='g', lw=3)
        plt.show()


def test_real_data():
    albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5'
    
    hf = h5py.File(albbdh_merged_file, 'r')
    
    al = hf['vars']['AL_DH_BB'][:,0,:]
    dates = hf['meta']['ts_dates'][:]
    dates_formatted = np.array([datetime.datetime.fromtimestamp(x).strftime('%Y%m%d') for x in dates])
    point_names = hf['meta']['point_names'][:]

    ## Return full T array or not
    if 0:
        res = np.array([snht(x, True)[0] for x in al.T])
        print(res.shape)
        print(res.mean(axis=0).shape)
    else:
        res = np.array([snht(x, False) for x in al.T])
        print(res.shape)
        stat = res.T[0]
        nvalid = res.T[2]
        pval = compute_pval_from_cache(stat, nvalid)
        print("pval=", pval)

    ## Plot 
    if 0:
        fig, ax1 = plt.subplots(figsize=(30/2.54, 20/2.54)) 
        for r in res:
            ax1.plot(r, alpha=0.1, c='k')
        ax1.plot(res.mean(axis=0))
        if 0:
            nbins = np.linspace(0,1000,101)
            ist = np.apply_along_axis(lambda a: np.log10(np.histogram(a[~np.isnan(a)], bins=nbins)[0]), 0, res)
            ax1.imshow(hist, origin='lower')
        plt.tight_layout()
        fname = 'res_pval_realdata.png'
        plt.savefig(fname)
        os.system('feh '+fname)


def test_recursive_snht():
    """
    Usage for plotting:
    -------------------

    python TOOL_snht.py
    ./crop_merge.sh
    python TEST_pptx.py
    """

    # Real data
    if 1:
        albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5'
        #albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_lai_MERGED/timeseries_198125_202017.h5'
        #albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_fapar_MERGED/timeseries_198125_202017.h5'
        with h5py.File(albbdh_merged_file, 'r') as hf:
            start = 190
            end = 200
            al = hf['vars']['AL_DH_BB'][:,0,start:end]
            #al = hf['vars']['LAI'][:,0,start:end]
            #al = hf['vars']['fAPAR'][:,0,start:50]
            #al = hf['vars']['LAI'][:,0,start:50]
            #al = hf['vars']['AL_DH_BB'][:,0,start:200]

            point_names = hf['meta']['point_names'][start:end]
            point_names = [i.decode('utf8') for i in point_names]
            dates = hf['meta']['ts_dates'][:]
    ## Synthetic data
    else:
        y = [] 
        y.append(0.1*np.random.rand(250)+ 0.0)
        y.append(0.1*np.random.rand(250) + 0.5)
        y.append(0.1*np.random.rand(250) - 1.)
        y.append(0.1*np.random.rand(250) + 0.0 )
        y = np.concatenate(y)

    #source = 'cache_file'
    source = 'compute_snht'



    if source=='compute_snht':
        break_list = []
        
        for idy_rel,y in enumerate(al.T):
            idy = idy_rel+start
            print('---', idy)
        
            x = np.arange(len(y))

            #y = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y)) # normalize

            ## Compute snht recursively
            res_dic = recursive_snht_dict(x, y, dates=dates, parent='1.1', max_lvl=4, nb_year_min=5)
            #res_dic = {k:{**res_dic[k],'parent':k.split('>')}}
            # Compute parent and childs
            res_dic_sorted = sorted(res_dic.keys(), key=lambda x:len(x))
            res_dic = [res_dic[i] for i in res_dic_sorted]

            ## Compute child
            for ni in res_dic:
                for nj in res_dic:
                    if ni['name']==nj['parent']:
                        ni['child'].append(nj['name'])

            ## print some info
            for k,v in zip(res_dic_sorted, res_dic):
                print('{:8} | {:2} | {:2} | {}'.format(k, v['parent'], v['name'], v['child']))
            
            break_list.append(res_dic)
    
    elif source=='cache_file': 
        with open('break_list.json', 'r') as f:
            frozen = f.read()
    
    plot_breaks(break_list, dates, point_names, start)
    

    ## Flatten everything to format for a DataFrame
    flat_res = []
    for i in range(len(break_list)):
        for j in range(len(break_list[i])):
            if 'snht' in break_list[i][j].keys():
                flat_res.append({**break_list[i][j], **break_list[i][j]['snht'], 'point_name':point_names[i] })
                del(flat_res[-1]['snht'])
            else:
                flat_res.append({**break_list[i][j], 'point_name':point_names[i] })
    print(len(flat_res))    


    ## Create a dataframe
    df = pd.DataFrame(flat_res)
    df = df.set_index(['point_name', 'name'])
    df['n_len'] = pd.to_numeric(df['n_len'], downcast='integer')
    df['bp_date'] = pd.to_datetime(df['bp_date'], unit='s')
    #df['cp'] = pd.to_numeric(df['cp'], downcast='integer')
    #print(df.drop(columns=['x', 'y']))
    print(df.drop(columns=['x', 'y', 'child', 'parent']))

    df.to_pickle('res_snht_withxy.gzip')
    df = df.drop(columns=['x', 'y', 'child', 'parent'])
    df.to_csv('res_snht.csv', sep=';')
    df.to_pickle('res_snht_light.gzip')

    sys.exit()

    plt.clf()
    plt.hist(break_list.T[0], 200)
    #plt.hist(break_list.T[0], )
    plt.savefig('tmp/snht_hist.png')


def plot_breaks(break_list, dates, point_names, start, fig_size=(15/2.54, 4/2.54)):

    print('--- Plot graphs...')
    
    #from adjustText import adjust_text
    
    date_lim = [pd.to_datetime(dates[0], unit='s'), pd.to_datetime(dates[-1], unit='s')]
    year_frac = [year_fraction(d) for d in date_lim]

    for idy_rel,res_dic in enumerate(break_list):
        idy = idy_rel+start

        ## Plot result
        #fig, ax1 = plt.subplots()
        fig, ax1 = plt.subplots(figsize=fig_size) 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        col_cycle = cycle(colors[:4])

        texts_top = [] # store annotations to apply adjust_text afterwards
        texts_bottom = [] # store annotations to apply adjust_text afterwards

        #for k,v in res_dic.items():
        for v in res_dic:
            v['x'] = (year_frac[1]-year_frac[0]) * (v['x']/(len(dates)-1)) + year_frac[0]
            col = random.choice(colors)
            if 1:
                #if len(v['child'])>0:
                if v['status'] in ['valid']:
                    #ax1.plot([v['x'][v['snht']['cp']]]*2, [0, 1], c='k' ) # vertical lines
                    ax1.axvline(v['x'][v['snht']['cp']], ymin=0.15, ymax=0.85, c='k' ) # vertical lines
                    #ax1.plot([v['x'][v['snht']['cp']]]*2, [0, v['snht']['T']/res_dic[0]['snht']['T']*np.nanmax(res_dic[0]['y'])], c='k') # vertical lines
                    
                    ## Add annotations
                    trans = blended_transform_factory(x_transform=ax1.transData, y_transform=ax1.transAxes)
                    # T value at the top
                    if 0:
                        texts_top.append(ax1.annotate('{:.1f}'.format(v['snht']['T']),
                                     xy=(v['x'][v['snht']['cp']], 1.),  xycoords=trans,
                                     xytext=(v['x'][v['snht']['cp']], 1.), textcoords=trans, ha='center', va='top', fontsize=10) )
                    # break level at the botoom
                    texts_bottom.append(ax1.annotate('{:d}'.format(v['lvl']),
                                 xy=(v['x'][v['snht']['cp']], 0.),  xycoords=trans,
                                 xytext=(v['x'][v['snht']['cp']], 0.), textcoords=trans, ha='center', fontsize=10) )

            if 1:
                #if len(v['child'])==0:
                #if v['status'].startswith('stop:'):
                #if v['status'] in ['stop:lvl', 'stop:pval']:
                if v['status']=='stop:lvl':
                    #ax1.plot(v['x'], v['y'], c=next(col_cycle), lw=2) # actual data
                    ax1.plot(v['x'], v['y'], c='#1f77b4', lw=2) # actual data
                if v['status']=='stop:pval':
                    ax1.plot(v['x'], v['y'], c='#ff7f0e', lw=2) # actual data
                if v['status']=='stop:edge':
                    ax1.plot(v['x'], v['y'], c='#9467bd', lw=2) # actual data
                if v['status']=='stop:len':
                    ax1.plot(v['x'], v['y'], c='#2ca02c', ls=':') # actual data
                if v['status']=='stop:nan':
                    ax1.plot(v['x'], v['y'], c='#d62728', ls=':') # actual data
            
            ## Plot the value of the slope over the segment
            if not v['trend'].startswith('not_'):
                #print('{} - {} (#{})'.format(v['trend'], point_names[idy_rel], idy))
                idmean = round(0.5*(np.nanargmax(v['x'])+np.nanargmin(v['x'])))
                ax1.annotate(v['trend'], xy=(v['x'][idmean], v['y'][idmean]), xycoords='data',
                             xytext=(0, 10), textcoords='offset points', horizontalalignment='center')

        ## Write the minimum at the bottom left
        ax1.annotate('{:.2}'.format(np.nanmin(res_dic[0]['y'])),
                     xy=(0., 0.),  xycoords='axes fraction',
                     xytext=(0., 0.), textcoords='axes fraction')
        ## Write the maximum at the top left
        ax1.annotate('{:.2}'.format(np.nanmax(res_dic[0]['y'])),
                     xy=(0., 1.),  xycoords='axes fraction',
                     xytext=(0., 1.), textcoords='axes fraction', verticalalignment='top')
        ## Write the title
        ax1.annotate('{} (#{})'.format(point_names[idy_rel], idy),
                     xy=(0., 1.),  xycoords='axes fraction',
                     xytext=(0., 1.05), textcoords='axes fraction', verticalalignment='baseline')



        ## use adjust_text library to fix overlapping text
        #adjust_text(texts_top, autoalign='x', expand_text=(1.5,1.2), force_text=(0.5, 0.25), avoid_points=False, va='top')
        #adjust_text(texts_bottom, autoalign='x', expand_text=(1.5,1.2), force_text=(0.5, 0.25), avoid_points=False, va='center')

        ## Setup axis

        if 0:
            ax1.get_xaxis().set_ticks([0, len(dates)-1])
            #date_ticks = [d.strftime('%Y-%m-%d') for d in date_lim]
            #ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
            #ax1.set_yticklabels(date_ticks)
            date_ticks = ticker.FuncFormatter(lambda x, pos: pd.to_datetime(dates[int(x)], unit='s').strftime('%Y-%m'))
            ax1.xaxis.set_major_formatter(date_ticks)

        ax1.get_yaxis().set_ticks([])
        ax1.set_ylim(0.0,0.8)

        plt.tight_layout()
        plt.savefig('output_recursive_snht/output_snht_{:03d}.png'.format(idy))
        plt.close(fig)


def VITO_recursive_snht(cache_file_name, var_name, max_lvl=3, nan_threshold=0.3, nb_year_min=5, alpha=0.05, edge_buffer=0, plot_snht=True, plot_size=(15/2.54, 4/2.54)):

    b_deb = 0

    ## Load data
    ##----------
    if b_deb:
        input_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5'
        #input_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_lai_MERGED/timeseries_198125_202017.h5'
        #input_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_fapar_MERGED/timeseries_198125_202017.h5'
    else:
        input_file = cache_file_name

    with h5py.File(input_file, 'r') as hf:
        if b_deb:
            start = 190
            end = 200
            var = hf['vars']['AL_DH_BB'][:,0,start:end]
            #var = hf['vars']['LAI'][:,0,start:end]
            #var = hf['vars']['fAPAR'][:,0,start:end]
            point_names = [i.decode('utf8') for i in hf['meta']['point_names'][start:end]]
        else:
            var = hf['vars'][var_name][:,0,:]
            point_names = [i.decode('utf8') for i in hf['meta']['point_names'][:]]
            start=0

        dates = hf['meta']['ts_dates'][:]


    ## Run recursive snht test and store raw results
    ##----------------------------------------------

    break_list = []
    
    for idy_rel,y in enumerate(var.T):
        idy = idy_rel+start
        print('---', idy)
    
        x = np.arange(len(y))

        ## Compute snht recursively
        res_dic = recursive_snht_dict(x, y, dates=dates, parent='1.1', max_lvl=max_lvl, nan_threshold=nan_threshold, nb_year_min=nb_year_min, alpha=alpha, edge_buffer=edge_buffer)
        # Compute parent and childs
        res_dic_sorted = sorted(res_dic.keys(), key=lambda x:len(x))
        res_dic = [res_dic[i] for i in res_dic_sorted]

        ## Compute child
        for ni in res_dic:
            for nj in res_dic:
                if ni['name']==nj['parent']:
                    ni['child'].append(nj['name'])

        ## print some info
        for k,v in zip(res_dic_sorted, res_dic):
            print('{:8} | {:2} | {:2} | {}'.format(k, v['parent'], v['name'], v['child']))
        
        break_list.append(res_dic)
    

    ## Plot breaks
    ##----------------------------------- 
    apply_mk_test_on_valid_data(break_list)
      

    ## Plot breaks
    ##----------------------------------- 
    if plot_snht:
        plot_breaks(break_list, dates, point_names, start)


    ## Flatten and format for a DataFrame
    ##----------------------------------- 

    ## Flatten
    flat_res = []
    for i in range(len(break_list)):
        for j in range(len(break_list[i])):
            if 'snht' in break_list[i][j].keys():
                flat_res.append({**break_list[i][j], **break_list[i][j]['snht'], 'point_name':point_names[i] })
                del(flat_res[-1]['snht'])
            else:
                flat_res.append({**break_list[i][j], 'point_name':point_names[i] })

    ## Create a dataframe
    df = pd.DataFrame(flat_res)
    df = df.set_index(['point_name', 'name'])
    df['n_len'] = pd.to_numeric(df['n_len'], downcast='integer')
    df['bp_date'] = pd.to_datetime(df['bp_date'], unit='s')

    
    ## Write output file
    ##------------------

    input_file = pathlib.Path(input_file)
    output_path = pathlib.Path('./output_snht/') / input_file.parts[-2]
    # Make output dir if not exists
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / input_file.parts[-1].replace('.h5', '_SNHT_break_test_{}.csv'.format(var_name))
    output_pickle = output_path / input_file.parts[-1].replace('.h5', '_SNHT_break_test_{}.pickle.gzip'.format(var_name))
    
    # Output to pickle with full data
    df.to_pickle(output_pickle)
    # Drop some colums for csv
    #print(df.drop(columns=['x', 'y']))
    print(df.drop(columns=['x', 'y', 'child', 'parent']))
    df = df.drop(columns=['x', 'y', 'child', 'parent'])
    
    df.to_csv(output_csv, sep=';')
    print("--- SNHT break test result file saved to:\n{}".format(output_csv))


def recursive_snht_dict(x, y, dates=None, parent='1.1', max_lvl=3, nan_threshold=0.3, nb_year_min=5, alpha=0.05, edge_buffer=0):
    lvl = parent.count('/')+1
    if lvl==1:
        p = ''
        name = '1.1'
    else:
        p = parent.split('/')[-2]
        name = parent.split('/')[-1]
    res = {}
    n_nan = np.isnan(y).sum()/len(y)
    base = {'x':x, 'y':y, 'lvl':lvl, 'parent':p, 'name':name, 'child':[], 'nan[%]':n_nan*100, 'n_len':len(y)}
    if len(y)<=36*nb_year_min:
        print('Stop at node {} because data length is too short ({} instead of {} min)'.format(name, len(y), 36*nb_year_min))
        res[parent] = {'status':'stop:len', **base}
        return res
    elif n_nan>=nan_threshold:
        print('Stop at node {} because too much nan in data ( {:.2f}% instead of {}% max)'.format(name, 100*n_nan, 100*nan_threshold))
        res[parent] = {'status':'stop:nan', **base}
        return res
    elif lvl>max_lvl:
        print('Stop at node {} because max level has been reach ({} on {} max)'.format(name, lvl, max_lvl))
        res[parent] = {'status':'stop:lvl', **base}
        return res
    res[parent] = {'snht':fast_snht_test(y, alpha=alpha), **base}
    snht = res[parent]['snht']
    #print(snht)
    if (snht['h']):
        print(edge_buffer, snht['cp'])
        if edge_buffer < snht['cp'] < len(y)-edge_buffer:
            res[parent]['status'] = 'valid'
            if dates is not None:
                res[parent]['snht']['bp_date'] = dates[x[snht['cp']]]
            xa = x[:snht['cp']]
            xb = x[snht['cp']:]
            ya = y[:snht['cp']]
            yb = y[snht['cp']:]
            nloc = int(name.split('.')[-1])
            res = {**res,
                   **recursive_snht_dict(xa, ya, dates=dates, parent=parent+'/'+str(lvl+1)+'.'+str(nloc*2-1), max_lvl=max_lvl, nan_threshold=nan_threshold, nb_year_min=nb_year_min, alpha=alpha, edge_buffer=edge_buffer),
                   **recursive_snht_dict(xb, yb, dates=dates, parent=parent+'/'+str(lvl+1)+'.'+str(nloc*2), max_lvl=max_lvl, nan_threshold=nan_threshold, nb_year_min=nb_year_min, alpha=alpha, edge_buffer=edge_buffer) }
        else:
            print('Stop at node {} because break location is too close of the sides ( cp={:d} )'.format(name, snht['cp']))
            res[parent]['status'] = 'stop:edge'
    else:
        print('Stop at node {} because snht test not significant ( p={:.3f} for p_max={})'.format(name, snht['p'], alpha))
        res[parent]['status'] = 'stop:pval'
    return res 



def fast_snht_test(x, alpha=0.05):
    x = np.asarray(x)
    tmax, tloc, nvalid, nnan, mu1, mu2, mag = snht(x, False) 
    stat = tmax
    pval = compute_pval_from_cache_scalar(stat, int(nvalid))
    if pval<alpha:
        h = True
    else:
        h = False
    
    res_dic = {'h':h, 'cp':int(tloc), 'p':pval, 'T':tmax, 'mu1':mu1, 'mu2':mu2, 'mag[%]':mag}
    
    return res_dic

def apply_mk_test_on_valid_data(break_list):
    for idy_rel,res_dic in enumerate(break_list):
        for v in res_dic:
            if v['status'] in ['stop:lvl', 'stop:pval', 'stop:edge']:
                res_mk = mk_test(v['y'])
                if res_mk[0]<=0.05:
                    #print('ok', res_mk)
                    v['trend'] = '{:.3e}'.format(res_mk[2])
                else:
                    #print('xx', res_mk)
                    v['trend'] = 'not_significant'
            else:
                v['trend'] = 'not_computed'


def mk_test(y):
    n_zero = (y==0.0).sum()
    #print("{:.4f} / {} / {} / {}".format(np.isnan(y).sum()/len(y), np.nanmin(y), np.nanmax(y), n_zero))
    ## To avoid 0.0 tie that makes the sort algo of median fortran computation stuck in very long loops,
    ## we add very low fake noise to all 0.0 values to allow the sort algo to perform efficiently.
    y[y==0.0] = 1.e-6*np.random.rand(n_zero)
    return m.mk_trend(len(y), np.arange(len(y)), y, 3)
    
def debug_func():
    x = np.array([1,1,1,2,2,2])
    res = fast_snht_test(x)
    print(res)

def main():
    #test_fake_data()
    #test_real_data()
    #test_recursive_snht()

    #VITO_recursive_snht()
    VITO_recursive_snht('/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5', 'AL_DH_BB', max_lvl=4, nb_year_min=5, edge_buffer=36)

    #debug_func()

if __name__=='__main__':
    main()
