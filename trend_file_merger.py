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
    


def plot_trends(product, chunks, plot_name, plot_choice, scale_tendency, config, phash):
    """
    Faster version: load only required data, not the full disk each time.
    import xarray is still long (several second) but only at the first call
    """
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    

    input_trend_file = pathlib.Path(config['output_path']['trend']) / product / config['output_path']['merged_filename']
    output_path = pathlib.Path(config['output_path']['plot']) / product

    # Shortcut to [ylim1:ylim2,xlim1:xlim2]
    zone_bnd = chunks.get_limits('global', 'slice') 
    x1,x2,y1,y2 = chunks.get_limits('global', 'tuple')
    zone_bnd_ext = (slice(y1-1,y2+1), slice(x1-1,x2+1)) # extended zone to compute cell corners

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
    lat_MSG = 0.0001*latmsg['LAT'][zone_bnd_ext]
    lonmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG = 0.0001*lonmsg['LON'][zone_bnd_ext]
    

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

    cm = 'jet'
    #cm = 'nipy_spectral'
    #cm = 'RdBu_r'

    ## use pcolormesh directly with recreated mesh

    # Note:
    # The use of xarray yield warning since x, y and c have the same size and so nearest interpolation is used in pcolormesh.
    # This may cause plotting  problem when it tries to recreate cells when the projection is not monotonic (this is the case
    # here when trying to plot Euro for example).
    # One solution is to yield manually the cells coordinates to pcolormesh directly.
    # See: 
    # https://github.com/matplotlib/matplotlib/issues/18317#issuecomment-678666099
    # https://bairdlangenbrunner.github.io/python-for-climate-scientists/matplotlib/pcolormesh-grid-fix.html

    # Recreate cells coordinates for each pixel value
    # Compute the mean of the lon/lat of the 4 adjacent cells to have the new cell corner
    lon = lon_MSG
    lon = 0.25*(lon[:-1,:-1]+lon[1:,:-1]+lon[:-1,1:]+lon[1:,1:])
    lat = lat_MSG
    lat = 0.25*(lat[:-1,:-1]+lat[1:,:-1]+lat[:-1,1:]+lat[1:,1:])

    if plot_choice=='sn':
        trends[plot_choice][view_zenith>75] = np.NaN
        vn,vx = (-1E-1, 1E-1)

    elif plot_choice=='zval':
        vn, vx = (-10, 10)

    elif plot_choice=='pval':
        vn,vx = (0., 1.)

    elif plot_choice=='len':
        vn = np.nanmin(trends[plot_choice])
        vx = np.nanmax(trends[plot_choice])

    imm = ax.pcolormesh(lon, lat, trends[plot_choice], transform=ccrs.PlateCarree(), vmin=vn, vmax=vx, cmap=cm)

    cbar = plt.colorbar(imm, ax=ax, orientation='horizontal', shrink=.8, pad=0.05, aspect=10)
    cbar.ax.tick_params(labelsize=14, rotation=90)

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











   
