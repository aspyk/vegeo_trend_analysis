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

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 
import h5py # import after netCDF4 otherwise 'RuntimeError: NetCDF: HDF error' on VITO VM
from datetime import datetime
import os,sys
import re
import fnmatch
import traceback
from timeit import default_timer as timer
import generic
import pathlib
from tools import SimpleTimer


class TimeSeriesExtractor():
    def __init__(self, product, start, end, chunks, config, phash):
        self.start = start
        self.end = end
        self.product = product
        self.chunks = chunks
        self.hash = phash
        
        self.config = config[self.product]

        self.output_path = pathlib.Path(config['output_path']['extract'])
        (self.output_path / self.product).mkdir(parents=True, exist_ok=True)

        self.mode = self.config['mode']
        
        if self.config['freq']=='10D':
            # Every 10D means 3 snapshots a month and 36 snapshots a year
            print('INFO: frequency = 10D so start and end dates are trimmed to start and end of month respectively') 
            self.start = self.start.replace(day=1)
            self.end = self.end.replace(day=pd.Period(self.end, freq='D').days_in_month) # freq='xxx' is here just to avoid a bug
            self.dseries = pd.date_range(self.start, self.end, freq='D')
            self.dseries = self.dseries[[d in [1,2,3] for d in self.dseries.day]] # select only days 3, 13 and 23 by month
        else:
            self.dseries = pd.date_range(self.start, self.end, freq=self.config['freq'])

        self.source = self.config['source']

        
        
    def get_lw_mask(self):
        """ Get the MSG disk land mask (0: see, 1: land, 2: outside space, 3: river/lake)"""
        hlw = h5py.File('./input_ancillary/HDF5_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_201610171300','r')
        self.lwmsk = hlw['LWMASK'][self.chunks.get_limits('global', 'slice')]
        hlw.close()

    def DEPRECATED_extract_albedo(self, chunk, date, dateindex):
        files = {}
        files['mdal_nrt'] = self.product_files['mdal_nrt'][dateindex]
        files['mdal'] = self.product_files['mdal'][dateindex]
        
        ## Reading MDAL albedo
        if date.year>=2016:
            file_name = files['mdal_nrt']
            if date.year<=2018:
                file_name += '.h5'
        else:   
            file_name = files['mdal']
        print(file_name)

        ## Extract data zone
        try:
            fh5 = h5py.File(file_name,'r')
        except Exception:
            traceback.print_exc()
            print ('File not found, moving to next file, assigning NaN to extacted pixel.')
            return None
    
        albedo = fh5['AL-BB-DH'][chunk.get_limits('global', 'slice')] # array of int
        zage = fh5['Z_Age'][chunk.get_limits('global', 'slice')]
        fh5.close()
        
        ## debug
        if 1:
            #probe = (slice(400, 420, 10), slice(1900, 1920, 10))
            probe = (400, 1900)
            albedo = fh5['AL-BB-DH'][probe] # array of int
            zage = fh5['Z_Age'][probe]
            qf = fh5['Q-flag'][probe]
            return [albedo, zage, qf]


        ## remove invalid data
        albedo = np.where(albedo==-1, np.nan, albedo/10000.)            
        albedo = np.where(zage>0, np.nan, albedo)
        
        return albedo
   
    def DEPRECATED_extract_evapo(self, chunk, date, dateindex):
        file_name = self.product_files['mdal'][dateindex]
        
        print(file_name)
        
        ## Extract data zone
        try:
            fh5 = h5py.File(file_name,'r')
        except Exception:
            traceback.print_exc()
            print ('File not found, moving to next file, assigning NaN to extacted pixel.')
            return None


        var = fh5['METREF'][chunk.get_limits('global', 'slice')] # array of int
        #zage = fh5['Z_Age'][*chunk.global_lim]
        fh5.close()
        
        ## remove invalid data
        var = np.where(var==-8000, np.nan, var/100.)            
        #albedo = np.where(zage>0, np.nan, albedo)
        
        return var

    def plot_histogram(self, tseries):
        ## Initialize an array to store histogram stats
        nbins = 100
        res_h = np.zeros((tseries.shape[0], nbins))
        dmin, dmax = (np.nanmin(tseries), np.nanmax(tseries))
        bins = np.linspace(dmin, dmax, nbins+1)
        pct_nan = []
        
        for i,cut in enumerate(tseries):
            ## Fill histogram
            h = np.histogram(cut, bins=bins)[0]
            #res_h[i] = np.log10(h) # w/o scaling
            #res_h[i] = np.log10(h/(np.count_nonzero(~np.isnan(net_albedo)))) # scale by total nb of not nan
            res_h[i] = np.log10(h/h.max()) # scale by max

            ## Count nan
            pct_nan.append(100*np.count_nonzero(np.isnan(cut))/cut.size)
    
        ## Generate and save histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        #res_h[res_h==0.0] = np.nan
        
        # You can then convert these datetime.datetime objects to the correct
        # format for matplotlib to work with.
        xnum = mdates.date2num(self.dseries)
        xlims = [xnum[0], xnum[-1]]

        # plot histogram
        ax.imshow(res_h.T, aspect='auto', origin='lower', extent=(xlims[0], xlims[1], dmin, dmax))
        
        # plot nan count
        ax2.plot(xnum, pct_nan, c='k', lw=1)

        ax.grid()

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()

        ax.set_xlabel('date')
        ax.set_ylabel(self.product)
        ax2.set_ylabel('% NaN')
        ax2.set_ylim(0,100)
        plt.savefig('res_hist_{}_{}.png'.format(self.hash, self.product))
            
    def plot_image_series(self, tseries):
        dmin, dmax = (np.nanmin(tseries), np.nanmax(tseries))
        
        out_path = './output_imageseries/'
        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
        
        ## Generate and save histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        init = dmin+np.zeros_like(tseries[0])
        init[0,0] = dmax
        img = ax.imshow(init, aspect='auto')
        for i,cut in enumerate(tseries):
            img.set_data(cut)
            #plt.pause(0.1)
            
            ax.set_title('{} - {}'.format(self.product, self.dseries[i].isoformat()))
            plt.savefig(out_path+'/res{1}_{0:03d}_{2}.png'.format(i, self.hash, self.product))
 
    def get_product_files(self, mode='predict'):
        """
        Generator that yield recursively open H5 file between two dates

        Parameters
        ----------

        mode : string, 'predict' or 'walk'
            - 'predict' is used to load daily/hourly files with predictable filename
            - 'walk' is used to load not easily predictable names and so 'walk' in subdirs of roots
            to find relevant files.
        """
        if self.mode=='predict':
            for date in self.dseries:
                ## merge root dir with substituted template
                file_name = pathlib.Path(self.config['root']) / date.strftime(self.config['template'])
                print(file_name)
                # Note: the with/yield pattern should be checked to see if files are corectly closed
                try:
                    with h5py.File(file_name, 'r') as h5f:
                        yield date, h5f
                except Exception:
                    #traceback.print_exc()
                    print ('File not found, moving to next file, assigning NaN to extacted pixel.')
                    yield date, None
        
        elif self.mode=='walk':
            ## Get year min and year max from start and end
            ymin = self.start.year
            ymax = self.end.year

            #print(ymin, ymax)

            ## Find in the subdir all the *.nc files and save as [datetime, file path]
            print('Look for data files in:\n', self.config['root'])
            flist = []
            for y in range(ymin, ymax+1):
                for dp, dn, fn in os.walk((pathlib.Path(self.config['root'])/str(y)).as_posix()):
                    if len(fn)>0:
                        fl = fnmatch.filter(fn, '*.nc')
                        if len(fl)==1:
                            ncfile = fl[0]
                            # Check sensor type
                            if self.chunk.sensor not in ncfile:
                                continue
                        else:
                            print('WARNING: no .nc file found')
                            continue
                        # Use regex to catch the date in the filename:
                        # starts with '_' followed by a 1 or 2
                        # then between 7 and 13 digits and a final '_'
                        p = re.compile('_[12]\d{7,13}_')
                        m = p.search(ncfile)
                        date_str = m.group()[1:-1]
                        if len(date_str[8:])==6:
                            pattern = "%H%M%S"
                        elif len(date_str[8:])==4:
                            pattern = "%H%M"
                        date_obj = datetime.strptime(date_str, "%Y%m%d"+pattern)
                        flist.append([date_obj, pathlib.Path(dp) / ncfile])


            ## Process data get from walking with pandas

            df = pd.DataFrame(flist, columns=['datetime','path'])
            # Trim the dates using start and end in a DataFrame
            df = df.set_index('datetime') 
            df = df.sort_index().loc[self.start:self.end]
            # DEPRECATED: to be done in merge module: Remove duplicated date with VGT sensor
            #df = df[~((df.index.duplicated(keep=False)) & (df['sensor']=='VGT'))]
            #df = df.drop(columns=['sensor'])

            ## The time series may have missing data and we have to take them into account.
            ## self.dfseries is the list of all theoretical dates (ex d1 to d4) so we are going
            ## to construct a dataframe based on it and we are then join it with df that contains
            ## only valid data in order to pass from df = 
            ##  d2 path2
            ##  d3 path3
            ## to self.df_full =
            ##  d1 nan
            ##  d2 path2
            ##  d3 path3
            ##  d4 nan

            def test(date):
                """
                Apply a formated index to the three month observations
                xx, yy and zz with xx < yy < zz become 1, 2 and 3
                We use this because day number are not always the same, ex:
                - c3s albedo : 10, 20 and 28|29|30|31
                - c3s LAI: 3, 13, 21|23|24
                """
                if date.day < 11:
                    return date.replace(day=1)
                elif date.day < 21:
                    return date.replace(day=2)
                else:
                    return date.replace(day=3)

            # Create fake formated date and set it as index
            df['global_id'] = df.index.to_series().map(test)
            df = df.reset_index().set_index('global_id')

            # debug
            #df = df.loc["2019-05-01":"2019-07-31"]
            #print(df)

            # Create a new theoretical df based only on dseries as index
            self.df_full = pd.DataFrame(index=self.dseries)
            self.df_full.index.name = 'global_id'

            # Join with valid data, empty date will automatically be filled with NaN or NaT
            self.df_full = self.df_full.join(df)


            # Show summary of valid data by year (or month)
            #df = df.groupby([df.index.year, df.index.month]).count()
            sum = self.df_full.groupby([self.df_full.index.year]).count()
            print(sum)
            print("Total path:", sum['path'].sum())
            
            # datetime index back to column to be easily selected 
            self.df_full = self.df_full.reset_index() 

            # Add helper columns
            # global_id is a timestamp like starting at 0 from 1970-01-01 on a 36 based year
            # useful to merge different cache file or with // or % operator to have year or position in year
            fd = self.df_full['global_id']
            fd = pd.to_datetime(fd).apply(lambda t: (t.year-1970)*36 + (t.month-1)*3 + t.day-1)
            self.df_full['global_id'] = fd 
            #self.df_full = self.df_full.set_index('global_id')


            ## Loop on these files and yield
            
            #for row in df.to_dict(orient="records"):
            for t in self.df_full.itertuples():
                # Note: the with/yield pattern should be checked to see if files are corectly closed
                try:
                    self.b_h5py_r = 1
                    if self.b_h5py_r:
                        with h5py.File(t.path, 'r') as h5file:
                            self.h5f = h5file
                            yield t
                    else:
                        with netCDF4.Dataset(t.path, 'r') as h5file:
                            self.h5f = h5file
                            yield t
                except Exception:
                    #traceback.print_exc()
                    t2 = t._replace(path=None)
                    yield t2

    def _extract_points(self, var, dtype):
        """
        This version load all the dataset and extract location from memory

        Direct load may be faster until a certain number of point depending on the system.
        On VITO benchmark give a value of 150.
        
        Notes
        -----
        1/ Profiling
        1.1/ Timing is good if there are a lot of points and loading of data is not
        too long. Best way should be using hyperslabs in h5py low level api
        but there are some array manipulation do to after extraction: hyperslab
        blocks are sorted by lat then lon (to be efficiently read in the file)
        and so it may yield mixed data when regions share same lat or overlap.

        1.2/ On 1km dataset, profiling gives:
        h5py:
        data = self.h5f[var][:] --> ~2.4s
        netCDF4:
        data = self.h5f.variables[var][:] --> ~24s, ie 10x h5py, to be continued...

        2/ To plot netCDF file attributes
        #print(len(self.h5f.ncattrs()))
        try:
            print(self.h5f.getncattr('platform'))
        except:
            print('warning: no platform attribute')
        """
        chunk = self.chunk
            
        prod_chunk = np.zeros((chunk.dim[1], chunk.box_size, chunk.box_size), dtype=dtype)
            
        ti = SimpleTimer()
        # check the number of points to load
        if chunk.dim[1] > 150: # two-step load
            if self.b_h5py_r:
                data = self.h5f[var][:]
                ti('h5py_indirect load time for {}:'.format(var))
            else:
                data = self.h5f.variables[var][:]
                ti('netCDF4 load time for {}:'.format(var))
            
            for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                prod_chunk[ii] = data[s]
            del(data)
            ti('memory load time for {}:'.format(var))

        else: # direct load
            for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                prod_chunk[ii] = self.h5f[var][s]
            ti('h5py_direct load time for {}:'.format(var))
        
        return prod_chunk

    def _get_c3s_albedo_points(self):

        age_chunk = self._extract_points('AGE', np.int16)
        q_chunk = self._extract_points('QFLAG', np.uint8)
        
        d = {}
        for v in self.config['var']:
            prod_chunk = self._extract_points(v, np.int16)
            error_chunk = self._extract_points(v+'_ERR', np.int16)
        

            # Discard pixels in the analysis when:
            # - Fill value in AL_* (-32767)
            # - Outside valid range in AL_* ([0, 10000])
            # - QFLAG indicates ‘sea’ or ‘continental water’ (QFLAG bits 0-1) → Fill value in product
            # - QFLAG indicates the algorithm failed (QFLAG bit 7) → Fill value in product
            # Optionally, also the following thresholds can be considered:
            # - *_ERR > 0.2
            # - AGE > 30
            b_print_sel = 0
            # -9999 is used instead of np.nan because data is of type int
            discard = -9999
            prod_chunk[(prod_chunk<0) | (prod_chunk>10000)] = discard
            if b_print_sel: print(self._count_nan(prod_chunk))
            prod_chunk[~((q_chunk & 0b11) == 0b01)] = discard # if bit 1 and 0 are not land (= 01) set np.nan
            if b_print_sel: print(self._count_nan(prod_chunk))
            prod_chunk[(q_chunk & (1<<7)) == 1] = discard # if bit 7 == 1 set np.nan
            if b_print_sel: print(self._count_nan(prod_chunk))
            prod_chunk[error_chunk > 2000] = discard 
            if b_print_sel: print(self._count_nan(prod_chunk))
            prod_chunk[age_chunk > 30] = discard 
            if b_print_sel: print(self._count_nan(prod_chunk))
            #sys.exit()
            #prod_chunk = prod_chunk.

            ## Agregate data using quality flag
            # Count True in each window and keep only when 75% is ok
            # 1 for 1x1, 12 for 4x4 and 108 for 12x12
            if self.chunk.c3s_resol=='4km':
                threshold = 1
            if self.chunk.c3s_resol=='1km':
                threshold = 12
            if self.chunk.c3s_resol=='300m':
                threshold = 108
            q_mask = np.count_nonzero(~(prod_chunk==discard), axis=(1,2)) >= threshold
            print('valid : {0} / {1}'.format(np.count_nonzero(q_mask), self.chunk.dim[1]) )
            #print('invalid indices:')
            #print(np.where(q_mask==0)[0])
            d[v] = 1e-4 * prod_chunk
            d[v][ d[v]==discard*1e-4 ] = np.nan # replace discard value by nan
            d[v] = np.nanmean(d[v], axis=(1,2))
            d[v] = np.where(q_mask,  d[v], np.nan)
            d[v].reshape((1,-1)) # fake 2D array (1,nb_pts)

        return d

    def _count_nan(a):
        return np.count_nonzero(np.isnan(a))    

    def _count_notnan(a):
        return np.count_nonzero(~np.isnan(a))    

    def extract_product(self):
        chunk = self.chunk
        if self.source=='msg':
            prod_chunk = self.h5f[self.config['var']][chunk.get_limits('global', 'slice')] # array of int
        
        elif self.source=='c3s':
            
            if chunk.input=='box':
                chunk_range = (0, *chunk.get_limits('global', 'slice')) # array of int
                prod_chunk = self.h5f[self.config['var']][chunk_range] # array of int
            
            elif chunk.input=='points':
                # Save point name
                self.ascii_pointnames = [n.encode("ascii", "ignore") for n in chunk.site_coor['NAME']]
 
                ## DEBUG: Plot landval
                if 0:
                    print("DEBUG: print landval sites")
                    print('Load...')
                    data_in = self.h5f[self.config['var']][:]
                    #data = self.h5f[self.config['var']][0,::2,::2]
                    #data = h5f['retrieval_flag'][:]
                    data = data_in[0,::2,::2]
                    del(data_in)

                    print('Plot...')
                    ti = SimpleTimer()
                    ti('t01')
                    fig, ax = plt.subplots()
                    ti('t02')
                    
                    data = 1e-4*data
                    ti('t1')
                    data[data<0] = 0.0
                    ti('t2')
                    print(data.shape)
                    print(data.min(), data.max())
                    #data = data & 0xFC1 # See Appendix A of D3.3.8-v2.1_PUGS_CDR-ICDR_LAI_FAPAR_PROBAV_v2.0_PRODUCTS_v1.1.pdf
                    print(data.min(), data.max())
                    ti('t3')
                    
                    ptype = 2
                    # Add landval sites directly in array
                    if ptype==1:
                        for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                            data[s] = 2e3
                    ti('t4')
                    if 1:
                        cm = 'jet'
                        cm = 'gist_ncar'
                        #mat = ax.imshow(data[0,::2,::2], cmap=cm)
                        mat = ax.imshow(data, cmap=cm)
                        # create an axes on the right side of ax. The width of cax will be 5%
                        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(mat, cax=cax) 
                    
                    # Add landval sites with scatter
                    if ptype==2:
                        pts = (0.5*chunk.site_coor[['ilat', 'ilon']].values.T).astype(np.int32)
                        print(pts.shape)
                        ax.scatter(pts[1], pts[0])
                        

                    #plt.savefig('res_c3s.png')
                    plt.show()

                    sys.exit()
                ## END DEBUG

                prod_chunk = self._get_c3s_albedo_points()


        ## remove invalid data
        if self.product=='lai':
            prod_chunk = np.where(prod_chunk==-10, np.nan, prod_chunk/1000.)            

        return prod_chunk
    
    def _write_ts_chunk(self, chunk, tseries, date_ts=None, add_var=[]):
        """Write time series of the data for each master iteration"""
        write_file = self.output_path / self.product / (self.hash+'_timeseries_'+'_'.join(chunk.get_limits('global', 'str'))+'.nc')
        write_file = write_file.as_posix()
    
        ds = netCDF4.Dataset(write_file, 'w', format='NETCDF4')
        
        ds.createDimension('x', tseries.shape[0])
        ds.createDimension('y', tseries.shape[1])
        ds.createDimension('z', tseries.shape[2])
        ds.createVariable('time_series_chunk', np.float, ('x','y','z'), zlib=True)
        
        ds.variables['time_series_chunk'][:] = tseries
            
        if date_ts is not None:
            assert date_ts.shape[0]==tseries.shape[0]
            ds.createVariable('time_series_date', np.int64, ('x',), zlib=True)
            
            ds.variables['time_series_date'][:] = date_ts
        
        ## Add additional variables if any
        # Additional variable have to be given in a list as dict with the following keys:
        # 'name': string, name of the variable
        # 'data': numpy array, the actual data 
        # 'type': string, representing the data type
        if len(add_var)>0:
            for v in add_var:
                tdim = []
                for idd,d in enumerate(v['data'].shape):
                    dname = '{}_d{}'.format(v['name'], idd)
                    ds.createDimension(dname, d)
                    tdim.append(dname)
                tdim = tuple(tdim)

                var = ds.createVariable(v['name'], v['type'], tdim, zlib=True)
                var_data = v['data']
                var[:] = var_data
        
        ds.close()
        print(">>> Data chunk written to:", write_file)

    def _write_ts_chunk2(self, chunk, out_var):
        """
        Write time series of the data
        Variable have to be given in a list as dict with the following keys:
        'name': string, name of the variable
        'data': numpy array, the actual data 
        'type': string, representing the data type
        """
        write_file = self.output_path / self.product / (self.hash+'_timeseries_'+'_'.join(chunk.get_limits('global', 'str'))+'.nc')
        write_file = write_file.as_posix()
    
        ds = netCDF4.Dataset(write_file, 'w', format='NETCDF4')
        
        if len(out_var)>0:
            for v in out_var:
                tdim = []
                for idd,d in enumerate(v['data'].shape):
                    dname = '{}_d{}'.format(v['name'], idd)
                    ds.createDimension(dname, d)
                    tdim.append(dname)
                tdim = tuple(tdim)

                var = ds.createVariable(v['name'], v['type'], tdim, zlib=True)
                var_data = v['data']
                var[:] = var_data
        
        ds.close()
        print(">>> Data chunk written to:", write_file)
        

    def run(self):
        b_use_mask = 0

        if b_use_mask: self.get_lw_mask()
       
        ## Loop on chunks
        for chunk in self.chunks.list:
            self.chunk = chunk

            ## Print some info
            if chunk.input=='box':
                print('***', 'SIZE (y,x)=(row,col)=({},{})'.format(*chunk.dim), 'GLOBAL_LOCATION=[{}:{},{}:{}]'.format(*chunk.get_limits('global', 'str')))
            elif chunk.input=='points':
                print('***', 'SIZE {} points'.format(chunk.dim[1]))


            ## Initialize time series with array of nan
            data_ts = {}
            for v in self.config['var']:
                data_ts[v] = np.full([len(self.dseries), *chunk.dim], np.nan)
            time_ts = []
            print('data shape:', data_ts[self.config['var'][0]].shape)

    
            ## Create the chunk mask
            if b_use_mask: 
                lwmsk_chunk = self.lwmsk[chunk.get_limits('local', 'slice')]
                ocean = np.where(lwmsk_chunk==0)
                land = np.where(lwmsk_chunk==1)
    
            res = [] # debug

            t0 = datetime.now()

            ## Loop on files
            for f in self.get_product_files():
                print("####################################################")
                print("{}/{} - {}".format(f.Index, len(self.dseries), f.datetime))
                if f.path is not None:
                    print(f.path.name)
                    tmp = self.extract_product()
                    for v in self.config['var']:
                        data_ts[v][f.Index] = tmp[v]
                    time_ts.append(f.datetime)
                else:
                    print ('File not found, moving to next file, assigning NaN to extacted pixel.')
                    time_ts.append(datetime(1970,1,1)) # set timestamp to 0 (=1970-01-01) if no data

                now = datetime.now()
                print('{} | elapsed: {}'.format(now, now-t0), flush=True)


            time_ts = np.array([np.datetime64(d).astype('<M8[s]') for d in time_ts])
            time_ts = time_ts.astype(np.int64)
            #print(time_ts)
            ## To store dates
            #bla = np.datetime64(datetime.datetime(2018,1,1)).astype('<M8[s]')
            #np.int64(4290000000).astype('<M8[s]')
            #datetime.datetime(1970,1,1, tzinfo=datetime.timezone.utc).timestamp() == 0


            # debug
            if 0:
                import matplotlib.pyplot as plt
                res = np.array(res)
                print(res.shape)
                plt.plot(res)
                plt.savefig('comp_age_qflag.png')
                return


            ## Try with a more generic writer
            if 0:
                add_var = []
                add_var.append({'type':'S2', 'name':'point_names', 'data':np.array(chunk.site_coor['NAME'])})
                self._write_ts_chunk(chunk, data_ts, time_ts, add_var)
            else:
                out_var = []
                for name, data in data_ts.items():
                    out_var.append({'type':np.float, 'name':name, 'data':data})
                out_var.append({'type':np.int64, 'name':'ts_dates', 'data':time_ts})
                out_var.append({'type':'S2', 'name':'point_names', 'data':np.array(chunk.site_coor['NAME'])})
                out_var.append({'type':np.uint16, 'name':'global_id', 'data':self.df_full['global_id'].values})
                self._write_ts_chunk2(chunk, out_var)

            #self.plot_histogram(tseries)
            #self.plot_histogram(tseries)
            #self.plot_image_series(tseries)
   
            del data_ts       
    

