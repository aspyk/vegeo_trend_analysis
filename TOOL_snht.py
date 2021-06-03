import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from adjustText import adjust_text
import h5py
import pandas as pd
from scipy import interpolate
from scipy import optimize
from datetime import *
from itertools import cycle

from timeit import default_timer as timer
import os,sys

import random
import pprint
import jsonpickle

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

    tloc = T.argmax()

    mu1 = x[:tloc].mean()
    mu2 = x[tloc:].mean()

    if return_array:
        # Duplicate first value of T to fit the shape of valid data
        Tarr = np.full(xin.shape, np.nan)
        Tarr[nan_mask] = np.concatenate([T[:1],T])
        return Tarr, nvalid, nnan
    else:
        return T.max(), idx[tloc], nvalid, nnan, mu1, mu2

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


def test_fake_data():
    x = []
    x.append(5.1*np.random.rand(250)+ 0.0)
    x.append(5.1*np.random.rand(250) + 0.5)
    # x.append(0.1*np.random.rand(300) - 1.)
    # x.append(0.1*np.random.rand(250) + 0.0 )
    x = np.concatenate(x)

    ti = SimpleTimer()

    [snht(x) for i in range(1000)]

    ti('end')

    t, tmax, tmaxloc , nvalid, nnan= snht(x, return_array=True)
    print(tmax, tmaxloc)

    if 1:
        ## Compute normal reference for p value 
        nmin = 10
        nmax= 1600
        sim = 20000

        mc_cache = np.empty((nmax-nmin, sim))
        res_root = []
        for n in range(nmin, nmax):

            t0 = datetime.now()
            res = mc_p_value(n, sim)[:,0]
            print(n, datetime.now()-t0)
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

        if 0:
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
    dates_formatted = np.array([datetime.fromtimestamp(x).strftime('%Y%m%d') for x in dates])
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

    # Real data
    if 1:
        #albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5'
        albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_lai_MERGED/timeseries_198125_202017.h5'
        #albbdh_merged_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_fapar_MERGED/timeseries_198125_202017.h5'
        with h5py.File(albbdh_merged_file, 'r') as hf:
            start = 0
            end = 50
            #al = hf['vars']['AL_DH_BB'][:,0,start:500]
            al = hf['vars']['LAI'][:,0,start:end]
            #al = hf['vars']['fAPAR'][:,0,start:50]
            #al = hf['vars']['LAI'][:,0,start:50]
            #al = hf['vars']['AL_DH_BB'][:,0,start:200]

            point_names = hf['meta']['point_names'][start:end]
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
            res_dic = recursive_snht_dict(x, y, parent='0.0', max_lvl=5, nb_year_min=5)
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
            break_list = jsonpickle.decode(frozen)
    
    print('--- Plot graphs...')
    for idy_rel,res_dic in enumerate(break_list):
        idy = idy_rel+start

        ## Plot result
        #fig, ax1 = plt.subplots()
        #fig, ax1 = plt.subplots(figsize=(10/2.54, 4/2.54)) 
        fig, ax1 = plt.subplots(figsize=(15/2.54, 3/2.54)) 
        #fig, ax1 = plt.subplots(figsize=(200/2.54, 10/2.54)) 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        col_cycle = cycle(colors[:4])

        texts_top = [] # store annotations to apply adjust_text afterwards
        texts_bottom = [] # store annotations to apply adjust_text afterwards

        #for k,v in res_dic.items():
        for v in res_dic:
            col = random.choice(colors)
            #cmap = matplotlib.cm.get_cmap('hsv')
            #col = '#'+''.join(['{:02x}'.format(i) for i in np.round(255*np.array(cmap(random.random())[:3])).astype(int)])
            if 1:
                #ax1.plot(v['x'], v['y']+v['lvl'], c=next(col_cycle)) # actual data
                #ax1.plot([v['x'][v['snht']['cp']]]*2, [0+v['lvl'], 1+v['lvl']], c='k') # vertical lines
                #ax1.plot(v['x'], v['y'], c=next(col_cycle)) # actual data
                #if len(v['child'])>0:
                if v['status'] in ['valid']:
                    #break_list.append([v['x'][v['snht']['cp']], v['snht']['T'], v['lvl']])
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
                #if v['name']=='0':
                    #ax1.plot(v['x'], v['y'], c='k', ls=':') # actual data
                    #ax1.plot(v['x'], v['y'], c='0.7') # actual data
            if 1:
                #if len(v['child'])==0:
                #if v['status'].startswith('stop:'):
                #if v['status'] in ['stop:lvl', 'stop:pval']:
                if v['status']=='stop:lvl':
                    #ax1.plot(v['x'], v['y'], c=next(col_cycle), lw=2) # actual data
                    ax1.plot(v['x'], v['y'], c='#1f77b4', lw=2) # actual data
                if v['status']=='stop:pval':
                    ax1.plot(v['x'], v['y'], c='#ff7f0e', lw=2) # actual data
                if v['status']=='stop:side':
                    ax1.plot(v['x'], v['y'], c='#9467bd', lw=2) # actual data
                if v['status']=='stop:len':
                    ax1.plot(v['x'], v['y'], c='#2ca02c', ls=':') # actual data
                if v['status']=='stop:nan':
                    ax1.plot(v['x'], v['y'], c='#d62728', ls=':') # actual data
        ax1.annotate('{:.2}'.format(np.nanmin(res_dic[0]['y'])),
                     xy=(0., 0.),  xycoords='axes fraction',
                     xytext=(0., 0.), textcoords='axes fraction')
        ax1.annotate('{:.2}'.format(np.nanmax(res_dic[0]['y'])),
                     xy=(0., 1.),  xycoords='axes fraction',
                     xytext=(0., 1.), textcoords='axes fraction', verticalalignment='top')

        ## use adjust_text library to fix overlapping text
        #adjust_text(texts_top, autoalign='x', expand_text=(1.5,1.2), force_text=(0.5, 0.25), avoid_points=False, va='top')
        #adjust_text(texts_bottom, autoalign='x', expand_text=(1.5,1.2), force_text=(0.5, 0.25), avoid_points=False, va='center')

        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        plt.tight_layout()
        plt.savefig('output_recursive_snht/output_snht_{:03d}.png'.format(idy))
        plt.close(fig)

    ## Save a json cache file
    if source=='compute_snht':
        frozen = jsonpickle.encode(break_list)
        with open('break_list.json', 'w') as f:
            f.write(frozen)
    
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
    print(df.drop(columns=['x', 'y']))

    sys.exit()

    plt.clf()
    plt.hist(break_list.T[0], 200)
    #plt.hist(break_list.T[0], )
    plt.savefig('tmp/snht_hist.png')

def recursive_snht_dict(x, y, parent='0', max_lvl=3, nan_threshold=0.3, nb_year_min=5, alpha=0.05, edge_buffer=0):
    lvl = parent.count('/')
    if lvl==0:
        p = ''
        name = '0.0'
    else:
        p = parent.split('/')[-2]
        name = parent.split('/')[-1]
    res = {}
    n_nan = np.isnan(y).sum()/len(y)
    base = {'x':x, 'y':y, 'lvl':lvl, 'parent':p, 'name':name, 'child':[], 'n_nan':n_nan}
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
        if edge_buffer < snht['cp'] < len(y)-edge_buffer:
            res[parent]['status'] = 'valid'
            xa = x[:snht['cp']]
            xb = x[snht['cp']:]
            ya = y[:snht['cp']]
            yb = y[snht['cp']:]
            nloc = int(name.split('.')[-1])
            res = {**res,
                   **recursive_snht_dict(xa, ya, parent=parent+'/'+str(lvl+1)+'.'+str(nloc*2), max_lvl=max_lvl, nan_threshold=nan_threshold, nb_year_min=nb_year_min, alpha=alpha),
                   **recursive_snht_dict(xb, yb, parent=parent+'/'+str(lvl+1)+'.'+str(nloc*2+1), max_lvl=max_lvl, nan_threshold=nan_threshold, nb_year_min=nb_year_min, alpha=alpha) }
        else:
            print('Stop at node {} because break location is too close of the sides ( cp={:d} )'.format(name, snht['cp']))
            res[parent]['status'] = 'stop:side'
    else:
        print('Stop at node {} because snht test not significant ( p={:.3f} for p_max={})'.format(name, snht['p'], alpha))
        res[parent]['status'] = 'stop:pval'
    return res


def recursive_snht(x, y, ax, parent='root', max_lvl=3, nan_threshold=0.3, nb_year_min=5):
    idr = '{:02x}'.format(random.getrandbits(8))
    print('start {} (parent {})'.format(idr, parent))
    col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lvl = parent.count('>')
    if 0:
        ax.plot(x, y+lvl, c=col[lvl])
    else:
        col = random.choice(col)
        #cmap = matplotlib.cm.get_cmap('hsv')
        #col = '#'+''.join(['{:02x}'.format(i) for i in np.round(255*np.array(cmap(random.random())[:3])).astype(int)])
        ax.plot(x, y+lvl, c=col)
    res = []
    res.append(parent)
    res.append(fast_snht_test(y, alpha=0.05))
    print(res[-1])
    #input()
    print(res[-1].h, len(y), (~np.isnan(y)).sum()/len(y))
    if (res[-1].h) and (len(y)>=36*nb_year_min) and (((~np.isnan(y)).sum()/len(y))>=(1-nan_threshold)) and (lvl<max_lvl):
        ax.plot([x[res[-1].cp]]*2, [1+lvl, 2+lvl], c='r')
        xa = x[:res[-1].cp]
        xb = x[res[-1].cp:]
        ya = y[:res[-1].cp]
        yb = y[res[-1].cp:]
        #res.append([recursive_snht(xa, ya, ax, parent=parent+'>'+idr), recursive_snht(xb, yb, ax, parent=parent+'>'+idr)])
        res.append([recursive_snht(xa, ya, ax, parent=parent+'>'+str(lvl+1)+'a'), recursive_snht(xb, yb, ax, parent=parent+'>'+str(lvl+1)+'b')])
    print('end', idr)
    return res


def fast_snht_test(x, alpha=0.05):
    x = np.asarray(x)
    tmax, tloc, nvalid, nnan, mu1, mu2 = snht(x, False) 
    stat = tmax
    pval = compute_pval_from_cache_scalar(stat, int(nvalid))
    if pval<alpha:
        h = True
    else:
        h = False
    
    res_dic = {'h':h, 'cp':int(tloc), 'p':pval, 'T':tmax, 'mu1':mu1, 'mu2':mu2}
    
    return res_dic

def main():
    #test_fake_data()
    #test_real_data()
    test_recursive_snht()

if __name__=='__main__':
    main()
