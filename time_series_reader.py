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
from netCDF4 import Dataset
import h5py # import after netCDF4 otherwise 'RuntimeError: NetCDF: HDF error' on VITO VM
from datetime import datetime
import os,sys
import re
import fnmatch
import traceback
from timeit import default_timer as timer
import generic
import pathlib


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
        
        self.dseries = pd.date_range(self.start, self.end, freq=self.config['freq'])

        self.source = self.config['source']
        

    def get_lw_mask(self):
        """ Get the MSG disk land mask (0: see, 1: land, 2: outside space, 3: river/lake)"""
        hlw = h5py.File('./input_ancillary/HDF5_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_201610171300','r')
        self.lwmsk = hlw['LWMASK'][self.chunks.get_limits('global', 'slice')]
        hlw.close()
    

    def extract_albedo(self, chunk, date, dateindex):
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
   
    def extract_evapo(self, chunk, date, dateindex):
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
 
    def _write_ts_chunk(self, chunk, tseries):
        """Write time series of the data for each master iteration"""
        write_file = self.output_path / self.product / (self.hash+'_timeseries_'+'_'.join(chunk.get_limits('global', 'str'))+'.nc')
        write_file = write_file.as_posix()
    
        nc_iter = Dataset(write_file, 'w', format='NETCDF4')
        
        nc_iter.createDimension('x', tseries.shape[0])
        nc_iter.createDimension('y', tseries.shape[1])
        if chunk.input=='box':
            nc_iter.createDimension('z', tseries.shape[2])
            nc_iter.createVariable('time_series_chunk', np.float, ('x','y','z'), zlib=True)
        else:
            nc_iter.createVariable('time_series_chunk', np.float, ('x','y'), zlib=True)
        
        nc_iter.variables['time_series_chunk'][:] = tseries
            
        nc_iter.close()
    
        print(">>> Data chunk written to:", write_file)

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
                        yield h5f
                except Exception:
                    #traceback.print_exc()
                    print ('File not found, moving to next file, assigning NaN to extacted pixel.')
                    yield None
        
        elif self.mode=='walk':
            ## Get year min and year max from start and end
            ymin = self.start.year
            ymax = self.end.year

            print(ymin, ymax)

            ## Find in the subdir all the *.nc files and save as [datetime, file path]
            flist = []
            for y in range(ymin, ymax+1):
                for dp, dn, fn in os.walk((pathlib.Path(self.config['root'])/str(y)).as_posix()):
                    if len(fn)>0:
                        ncfile = fnmatch.filter(fn, '*.nc')[0]
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

            ## Trim the dates using start and end in a DataFrame
            df = pd.DataFrame(flist, columns=['datetime','path'])
            df = df.set_index('datetime') 
            df = df.sort_index().loc[self.start:self.end]

            ## Loop on these files and yield
            for file_name in df['path']:
                print(file_name)
                # Note: the with/yield pattern should be checked to see if files are corectly closed
                try:
                    with h5py.File(file_name, 'r') as h5f:
                        yield h5f
                except Exception:
                    #traceback.print_exc()
                    print ('File not found, moving to next file, assigning NaN to extacted pixel.')
                    yield None


    def extract_product(self, h5f, chunk):
        if self.source=='msg':
            prod_chunk = h5f[self.config['var']][chunk.get_limits('global', 'slice')] # array of int
        
        elif self.source=='c3s':
            
            if chunk.input=='box':
                chunk_range = (0, *chunk.get_limits('global', 'slice')) # array of int
                prod_chunk = h5f[self.config['var']][chunk_range] # array of int
            
            elif chunk.input=='points':
                # Product
                data = h5f[self.config['var']][:]
                #data = h5f['retrieval_flag'][:]
                prod_chunk = np.zeros((*chunk.dim, 2*chunk.box_offset, 2*chunk.box_offset), dtype=np.uint16)
                for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                    prod_chunk[ii] = data[s]
                del(data)


                ## DEBUG: Plot landval
                if 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    
                    #data = data.astype(np.float)
                    print(data.min(), data.max())
                    #data = data & 0xFC1
                    print(data.min(), data.max())
                    for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                        data[s] = 2e3
                    cm = 'jet'
                    cm = 'gist_ncar'
                    mat = ax.imshow(data[0,::2,::2], cmap=cm)
                    #mat = ax.imshow(data[0], cmap=cm)
                    # create an axes on the right side of ax. The width of cax will be 5%
                    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(mat, cax=cax) 

                    #plt.savefig('res_c3s.png')
                    plt.show()

                    sys.exit()
                    ## END DEBUG

                # Quality
                data = h5f['retrieval_flag'][:]
                q_chunk = np.zeros((*chunk.dim,4,4), dtype=np.uint32)
                for ii,s in enumerate(chunk.get_limits('global', 'slice')):
                    q_chunk[ii] = data[s]
                del(data)
                ## Agregate data using quality flag
                # TODO: if more than 75% of the matrix is ok agregate

                # See Appendix A of D3.3.8-v2.1_PUGS_CDR-ICDR_LAI_FAPAR_PROBAV_v2.0_PRODUCTS_v1.1.pdf
                q_mask = (q_chunk & 0xFC1 ) == 0
                # Count True in each window and keep only when 75% is ok
                q_mask = np.count_nonzero(q_mask, axis=(1,2)) >= 12  # 12 for 4x4 windows and 108 for 12x12 
                print('non_zero q_mask: {0} / {1}'.format(np.count_nonzero(q_mask), *chunk.dim) )
                prod_chunk = prod_chunk.mean(axis=(1,2))

        ## remove invalid data
        if self.product=='lai':
            prod_chunk = np.where(prod_chunk==-10, np.nan, prod_chunk/1000.)            

        return prod_chunk
    

    def run(self):
        b_use_mask = 0

        if b_use_mask: self.get_lw_mask()
       
        ## Loop on chunks
        for chunk in self.chunks.list:
            t0 = timer()
            if chunk.input=='box':
                print('***', 'SIZE (y,x)=(row,col)=({},{})'.format(*chunk.dim), 'GLOBAL_LOCATION=[{}:{},{}:{}]'.format(*chunk.get_limits('global', 'str')))
            elif chunk.input=='points':
                print('***', 'SIZE {} points'.format(*chunk.dim))

            ## Initialize an array series with nan
            tseries = np.full([len(self.dseries), *chunk.dim], np.nan)
            print(tseries.shape)
    
            ## Create the chunk mask
            if b_use_mask: 
                lwmsk_chunk = self.lwmsk[chunk.get_limits('local', 'slice')]
                ocean = np.where(lwmsk_chunk==0)
                land = np.where(lwmsk_chunk==1)
    
            res = [] # debug

            ## Loop on files
            if 1:
                for h5index, h5file in enumerate(self.get_product_files()):
                    if h5file is not None:
                        print(h5index, 'OK')
                        tseries[h5index] = self.extract_product(h5file, chunk)

            # debug
            if 0:
                import matplotlib.pyplot as plt
                res = np.array(res)
                print(res.shape)
                plt.plot(res)
                plt.savefig('comp_age_qflag.png')
                return


            print(timer()-t0)
    
            self._write_ts_chunk(chunk, tseries)
            #self.plot_histogram(tseries)
            #self.plot_image_series(tseries)
   
            del tseries       
    

