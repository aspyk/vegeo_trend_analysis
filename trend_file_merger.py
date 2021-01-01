#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:54:31 2019

@author: moparthys
"""

import numpy as np
from netCDF4 import Dataset
import re
import h5py
import os,sys
import fnmatch
import pathlib
from timeit import default_timer as timer

def pprinttable(rows, header, fmt):
    w = 9
    h_format = "{:>%i}"%w * (len(header))
    row_format = ''
    for f in fmt:
        row_format += '{:>%i%s}'%(w,f)
    print(h_format.format(*header))
    for row in rows:
        print(row_format.format(*row))

def merge_trends(product, chunks, config, phash):
    """
    profiling:
        Slower part of the code are the 4 lines: 
            nc_output.variables['xxx'][:] = temp_store_trend[:,:,x]
        This is du to big amount of data and their compression.
        Call only once so not problematic.
    """

    b_debug = 0

    input_path = pathlib.Path(config['output_path']['trend']) / product
    output_file = input_path / config['output_path']['merged_filename']

    print('*** MERGE TRENDS')
    nc_output = Dataset(output_file, 'w')
    xdim = nc_output.createDimension('X_dim',3712)
    ydim = nc_output.createDimension('Y_dim',3712)
    Vardim = nc_output.createDimension('Var',4)
    
    output_data = nc_output.createVariable('chunk_scores_p_val',  np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_z_val',  np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_Sn_val', np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=9)
    output_data = nc_output.createVariable('chunk_scores_length', np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    
    temp_store_trend = np.full((3712,3712,4), np.nan)
    
    if b_debug:
        tile_map_glob = np.full((3712,3712), np.nan)
        tile_map_loc = np.full(chunks.dim, np.nan)

    ## Filter the trend files with the case hash
    flist = os.listdir(input_path) 
    filenames = fnmatch.filter(flist, '{}_*'.format(phash))
    print("Found {} files to merge.".format(len(filenames)))
            
    for child_file in filenames:

        child_path = input_path / child_file

        print(child_path.as_posix())
        
        # Retrieve the global coordinates from the filename
        data = re.findall(r"[-+]?\d*\.\d+|\d+", child_file)
        data = data[-4:]
   
        COL1 = int(data[0])
        COL2 = int(data[1])
    
        ROW1 = int(data[2])
        ROW2 = int(data[3])
    
        #print(ROW1, ROW2, COL1, COL2)

        child_nc = Dataset(child_path, 'r')
    
        data_child_z = child_nc.variables['chunk_scores_z_val'][:].astype('f')
        data_child_p = child_nc.variables['chunk_scores_p_val'][:].astype('f')
        data_child_length = child_nc.variables['chunk_scores_length'][:].astype('f')
        data_child_sn = child_nc.variables['chunk_scores_Sn_val'][:].astype('f')
    
        child_nc.close()
        
        
        temp_store_trend[ROW1:ROW2,COL1:COL2,0] = data_child_p
        temp_store_trend[ROW1:ROW2,COL1:COL2,1] = data_child_z
        temp_store_trend[ROW1:ROW2,COL1:COL2,2] = data_child_sn
        temp_store_trend[ROW1:ROW2,COL1:COL2,3] = data_child_length

        ## Fill a tile map to debug
        if b_debug:
            tmp = 0.5*np.ones((ROW2-ROW1,COL2-COL1))
            tmp[:,:] = data_child_length
            tmp = np.where(tmp==0.0, 0.0, 0.5) # create a mask for valid data (~land)
            tmp[:,[0,-1]] = 1.0 # add border of the tile
            tmp[[0,-1],:] = 1.0 # idem
            tile_map_glob[ROW1:ROW2,COL1:COL2] = tmp
            offsetx = chunks.get_limits('global', 'tuple')[0]
            offsety = chunks.get_limits('global', 'tuple')[2]
            tile_map_loc[ROW1-offsety:ROW2-offsety,COL1-offsetx:COL2-offsetx] = tmp

            # DEBUG
            var = temp_store_trend[chunks.get_limits('global', 'slice')]
            size = var.shape[0]*var.shape[1]
            header = ['var', 'valid[%]', 'min', 'max']
            fmt = ['s','.1f','.3f','.3f']
            rows = []
            v = var[:,:,3] 
            rows.append(['len', 100.*(1.-np.count_nonzero(v==0.0)/size), np.nanmin(v), np.nanmax(v)])
            v = var[:,:,0] 
            rows.append(['pval', 100.*(1-np.count_nonzero(v>0.05)/size), np.nanmin(v), np.nanmax(v)])
            v = var[:,:,1] 
            rows.append(['zval', 100.*(1-np.count_nonzero(v==0.0)/size), np.nanmin(v), np.nanmax(v)])
            v = var[:,:,2] 
            rows.append(['sn', 100.*(1-np.count_nonzero(v==0.5)/size), np.nanmin(v), np.nanmax(v)])
            pprinttable(rows, header, fmt)

    nc_output.variables['chunk_scores_p_val'][:] = temp_store_trend[:,:,0]
    nc_output.variables['chunk_scores_z_val'][:] = temp_store_trend[:,:,1]
    nc_output.variables['chunk_scores_Sn_val'][:] = temp_store_trend[:,:,2]
    nc_output.variables['chunk_scores_length'][:] = temp_store_trend[:,:,3]
            
    nc_output.close() 

    print("Output file saved to: {}".format(output_file))

    if b_debug:
        print('Plot debug tile map...')
        
        import matplotlib.pyplot as plt
        
        plt.imshow(tile_map_glob)
        plt.savefig('tile_map_glob.png')
        plt.imshow(tile_map_loc)
        plt.savefig('tile_map_loc.png')
    

def plot_trends_OLD(product, chunks, plot_name, plot_choice, scale_tendency, config, phash):
    """deprecated version, too slow on loading data"""
    
    t0 = timer()

    import cartopy.crs as ccrs
    
    print('t000', timer()-t0)
    t0 = timer()

    import xarray as xr
    
    print('t00', timer()-t0)
    t0 = timer()

    input_trend_file = pathlib.Path(config['output_path']['trend']) / product / config['output_path']['merged_filename']
    output_path = pathlib.Path(config['output_path']['plot']) / product

    # Shortcut to [ylim1:ylim2,xlim1:xlim2]
    zone_bnd = chunks.get_limits('global', 'slice')

    # Make dir if not exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('*** PLOT TRENDS')
    
    ## Read merged data
    nc_output = Dataset(input_trend_file, 'r')

    print('t01', timer()-t0)
    t0 = timer()

    trends = {}
    trends['pval'] = nc_output.variables['chunk_scores_p_val'][:]
    trends['zval'] = nc_output.variables['chunk_scores_z_val'][:]
    trends['len'] = nc_output.variables['chunk_scores_length'][:]
    trends['sn'] = nc_output.variables['chunk_scores_Sn_val'][:]
    nc_output.close()
    
    print('t1', timer()-t0)
    t0 = timer()

    ## Read MSG geographical data
    work_dir_msg = './input_ancillary/'

    latmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    lat_MSG = 0.0001*latmsg['LAT'][:]

    lonmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG = 0.0001*lonmsg['LON'][:]
    
    print('t10', timer()-t0)
    t0 = timer()

    hangle_view_zen = h5py.File(work_dir_msg+'GEO_L1B-ANGLES-MSG2_2012-06-15T13-00-00_V1-00.hdf5','r')
    view_zenith = hangle_view_zen['View_Zenith'][:]
    view_zenith = np.array(view_zenith,dtype='float')
    view_zenith[view_zenith==65535] = np.NaN
    view_zenith = view_zenith*0.01
    
    print('t11', timer()-t0)
    t0 = timer()

    trends['sn'][trends['sn']==999] = np.NaN
    trends['sn'][trends['pval']>0.05] = np.NaN

    trends['zval'][trends['zval']==999] = np.NaN
    trends['zval'][trends['pval']>0.05] = np.NaN

    print('t2', timer()-t0)
    t0 = timer()

    ## Plot 
    proj = ccrs.Geostationary()
    #fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection':proj})
    fig, ax = plt.subplots(subplot_kw={'projection':proj})

    print('t20', timer()-t0)
    t0 = timer()


    if plot_choice=='sn':
        trends[plot_choice][view_zenith>75] = np.NaN
        dx = xr.DataArray(trends[plot_choice][zone_bnd]*scale_tendency, dims = ('y','x'))
        # The value of scale tendency depends on the product, daily or high frequency, for daily, the value is 365
    else:
        dx = xr.DataArray(trends[plot_choice][zone_bnd], dims = ('y','x'))
    
    dx.coords['lat'] = (('y', 'x'), lat_MSG[zone_bnd])
    dx.coords['lon'] = (('y', 'x'), lon_MSG[zone_bnd])

    print('t21', timer()-t0)
    t0 = timer()

    cm = 'jet'
    #cm = 'nipy_spectral'
    #cm = 'RdBu_r'

    if plot_choice=='sn':
        #imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), vmin=-1E-5, vmax=1E-5, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), vmin=-1E-1, vmax=1E-1, cmap=cm, add_colorbar=False, extend='neither', ax=ax)


    if plot_choice=='zval':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap=cm, add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='pval':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=0., vmax=1., cmap=cm, add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='len':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cm, add_colorbar=False, extend='neither', ax=ax)


    print('t22', timer()-t0)
    t0 = timer()

    cbar = plt.colorbar(imm,ax=ax,orientation='horizontal',shrink=.8,pad=0.05,aspect=10)
    cbar.ax.tick_params(labelsize=14,rotation=90)

    ax.coastlines()
    plt.axis('off')
    ax.set_title('{} for {}'.format(plot_choice, plot_name.split(':')[1]), fontsize=20)
    
    zone_str = '_'.join(chunks.get_limits('global','str'))
    plot_fname = phash+'_'+plot_choice+'_'+zone_str+'.png'
    plot_path = output_path / plot_fname
    
    print('t3', timer()-t0)
    t0 = timer()

    #fig.savefig(plot_string_save, dpi=300)
    fig.savefig(plot_path, dpi=100)
    plt.close()   

    print('*** IMAGE SAVED >', plot_path.as_posix())

    print('t4', timer()-t0)
    t0 = timer()





def plot_trends(product, chunks, plot_name, plot_choice, scale_tendency, config, phash):
    """
    Faster version: load only required data, not the full disk each time.
    import xarray is still long (several second) but only at the first call
    """
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import xarray as xr
    
    input_trend_file = pathlib.Path(config['output_path']['trend']) / product / config['output_path']['merged_filename']
    output_path = pathlib.Path(config['output_path']['plot']) / product

    # Shortcut to [ylim1:ylim2,xlim1:xlim2]
    zone_bnd = chunks.get_limits('global', 'slice')

    # Make dir if not exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('*** PLOT TRENDS')
    
    ## Read merged data
    nc_output = Dataset(input_trend_file, 'r')

    trends = {}
    trends['pval'] = nc_output.variables['chunk_scores_p_val'][zone_bnd]
    trends['zval'] = nc_output.variables['chunk_scores_z_val'][zone_bnd]
    trends['len'] = nc_output.variables['chunk_scores_length'][zone_bnd]
    trends['sn']  = nc_output.variables['chunk_scores_Sn_val'][zone_bnd]
    nc_output.close()
    

    ## Read MSG geographical data
    work_dir_msg = './input_ancillary/'

    latmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    lat_MSG = 0.0001*latmsg['LAT'][zone_bnd]

    lonmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG = 0.0001*lonmsg['LON'][zone_bnd]
    
    hangle_view_zen = h5py.File(work_dir_msg+'GEO_L1B-ANGLES-MSG2_2012-06-15T13-00-00_V1-00.hdf5','r')
    view_zenith = hangle_view_zen['View_Zenith'][zone_bnd]
    view_zenith = np.array(view_zenith,dtype='float')
    view_zenith[view_zenith==65535] = np.NaN
    view_zenith = view_zenith*0.01
    
    trends['sn'][trends['sn']==999] = np.NaN
    trends['sn'][trends['pval']>0.05] = np.NaN

    trends['zval'][trends['zval']==999] = np.NaN
    trends['zval'][trends['pval']>0.05] = np.NaN


    ## Plot 
    proj = ccrs.Geostationary()
    #fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection':proj})
    fig, ax = plt.subplots(subplot_kw={'projection':proj})

    if plot_choice=='sn':
        trends[plot_choice][view_zenith>75] = np.NaN
        dx = xr.DataArray(trends[plot_choice]*scale_tendency, dims = ('y','x'))
        # The value of scale tendency depends on the product, daily or high frequency, for daily, the value is 365
    else:
        dx = xr.DataArray(trends[plot_choice], dims = ('y','x'))
    
    dx.coords['lat'] = (('y', 'x'), lat_MSG)
    dx.coords['lon'] = (('y', 'x'), lon_MSG)

    cm = 'jet'
    #cm = 'nipy_spectral'
    #cm = 'RdBu_r'

    if plot_choice=='sn':
        #imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), vmin=-1E-5, vmax=1E-5, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), vmin=-1E-1, vmax=1E-1, cmap=cm, add_colorbar=False, extend='neither', ax=ax)


    if plot_choice=='zval':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap=cm, add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='pval':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=0., vmax=1., cmap=cm, add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='len':
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cm, add_colorbar=False, extend='neither', ax=ax)

    cbar = plt.colorbar(imm,ax=ax,orientation='horizontal',shrink=.8,pad=0.05,aspect=10)
    cbar.ax.tick_params(labelsize=14,rotation=90)

    ax.coastlines()
    plt.axis('off')
    ax.set_title('{} for {}'.format(plot_choice, plot_name.split(':')[1]), fontsize=20)
    
    zone_str = '_'.join(chunks.get_limits('global','str'))
    plot_fname = phash+'_'+plot_choice+'_'+zone_str+'.png'
    plot_path = output_path / plot_fname
    
    #fig.savefig(plot_string_save, dpi=300)
    fig.savefig(plot_path, dpi=100)
    plt.close()   

    print('*** IMAGE SAVED >', plot_path.as_posix())











   
