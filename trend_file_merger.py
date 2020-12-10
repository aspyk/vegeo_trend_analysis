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
import matplotlib.pyplot as plt
import os,sys
import fnmatch

def pprinttable(rows, header, fmt):
    w = 9
    h_format = "{:>%i}"%w * (len(header))
    row_format = ''
    for f in fmt:
        row_format += '{:>%i%s}'%(w,f)
    print(h_format.format(*header))
    for row in rows:
        print(row_format.format(*row))

def merge_trends(input_path, file_trend_name, chunks, phash):

    print('*** MERGE TRENDS')
    nc_output = Dataset(input_path+'/'+file_trend_name,'w')
    xdim = nc_output.createDimension('X_dim',3712)
    ydim = nc_output.createDimension('Y_dim',3712)
    Vardim = nc_output.createDimension('Var',4)
    
    print('load0')

    output_data = nc_output.createVariable('chunk_scores_p_val',  np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_z_val',  np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_Sn_val', np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=9)
    output_data = nc_output.createVariable('chunk_scores_length', np.float, ('Y_dim','X_dim'), zlib=True, least_significant_digit=3)
    
    temp_store_trend = np.empty([3712,3712,4])
    temp_store_trend[:] = np.NaN
    
    tile_map_glob = np.full((3712,3712), np.nan)
    tile_map_loc = np.full(chunks.dim, np.nan)

    ## Filter the trend files with the case hash
    flist = os.listdir(input_path) 
    filenames = fnmatch.filter(flist, '{}_*'.format(phash))
            
    for child_file in filenames:

        #child_file = matching_master[j][0:-1]
        
        print(input_path+'/'+child_file)
        
        data = re.findall(r"[-+]?\d*\.\d+|\d+", child_file)
        data = data[-4:]
   
        COL1 = int(data[0])
        COL2 = int(data[1])
    
        ROW1 = int(data[2])
        ROW2 = int(data[3])
    

        print(ROW1, ROW2, COL1, COL2)

        child_nc = Dataset(input_path+'/'+child_file, 'r')
    
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

    print('Plot debug tile map...')
    plt.imshow(tile_map_glob)
    plt.savefig('tile_map_glob.png')
    plt.imshow(tile_map_loc)
    plt.savefig('tile_map_loc.png')
    

def plot_trends(input_path, output_path, file_trend_name, xlim1, xlim2, ylim1, ylim2,
                plot_name, plot_choice, scale_tendency):
    
    import cartopy.crs as ccrs
    import xarray as xr
    
    cdpzn = input_path

    print('*** PLOT TRENDS')
    
    ## Read merged data
    nc_output = Dataset(cdpzn+'/'+file_trend_name,'r')

    trends = {}
    trends['pval'] = nc_output.variables['chunk_scores_p_val'][:]
    trends['zval'] = nc_output.variables['chunk_scores_z_val'][:]
    trends['len'] = nc_output.variables['chunk_scores_length'][:]
    trends['sn'] = nc_output.variables['chunk_scores_Sn_val'][:]
    nc_output.close()
    
    ## Read MSG geographical data
    work_dir_msg = './input_ancillary/'

    latmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    lat_MSG = 0.0001*latmsg['LAT'][:]

    lonmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG = 0.0001*lonmsg['LON'][:]
    
    hangle_view_zen = h5py.File(work_dir_msg+'GEO_L1B-ANGLES-MSG2_2012-06-15T13-00-00_V1-00.hdf5','r')
    view_zenith = hangle_view_zen['View_Zenith'][:]
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
        dx = xr.DataArray(trends[plot_choice][ylim1:ylim2,xlim1:xlim2]*scale_tendency, dims = ('y','x'))
        '''The value of scale tendency depends on the product, daily or high frequency, for daily, the value is 365 '''
    else:
        dx = xr.DataArray(trends[plot_choice][ylim1:ylim2,xlim1:xlim2], dims = ('y','x'))
    
    dx.coords['lat'] = (('y', 'x'), lat_MSG[ylim1:ylim2,xlim1:xlim2])
    dx.coords['lon'] = (('y', 'x'), lon_MSG[ylim1:ylim2,xlim1:xlim2])

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
    
    zone_str = '_'.join([str(i) for i in [xlim1,xlim2,ylim1,ylim2]])
    plot_save = plot_name.split(':')[0]+'_'+plot_choice+'_'+zone_str+'.png'
    plot_string_save = output_path + '/' + plot_save
    plot_string_save = os.path.normpath(plot_string_save)
    
    #fig.savefig(plot_string_save, dpi=300)
    fig.savefig(plot_string_save, dpi=100)
    plt.close()   

    print('*** IMAGE SAVED >', plot_string_save)





    










   
