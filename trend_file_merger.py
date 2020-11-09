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
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import os

def merge_trends(input_path, file_trend_name, xlim1, xlim2, ylim1, ylim2, nmaster):

    cdpzn = input_path
    
    print('*** MERGE TRENDS')
    nc_output = Dataset(cdpzn+'/'+file_trend_name,'w')
    xdim = nc_output.createDimension('X_dim',3712)
    ydim = nc_output.createDimension('Y_dim',3712)
    Vardim = nc_output.createDimension('Var',4)
    
    output_data = nc_output.createVariable('chunk_scores_p_val',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_z_val',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    output_data = nc_output.createVariable('chunk_scores_Sn_val',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=9)
    output_data = nc_output.createVariable('chunk_scores_length',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    
    temp_store_trend = np.empty([3712,3712,4])
    temp_store_trend[:] = np.NaN
    

    tile_map = np.zeros((3712,3712))

    
    row_chunk_main = np.arange(xlim1,xlim2,nmaster)
    col_chunk_main = np.arange(ylim1,ylim2,nmaster)
    '''we can use 1000 or less file size 500, but need to change the dependencies in code as described by the comments '''
    chunks_row_final = np.append(row_chunk_main, [xlim2], axis=0)
    chunks_col_final = np.append(col_chunk_main, [ylim2], axis=0)    

    nchild = 100
    chunks_main = np.arange(chunks_row_final[0],chunks_row_final[-1],nchild)
    chunks_final = np.append(chunks_main,[xlim2],axis=0)
       
        
    fp = open(input_path+'/'+'filelist.txt')
    
    filenames = fp.readlines()
    for i in range(len(chunks_final)-1):
        print (chunks_final[i], chunks_final[i+1])
            
        matching_master = [s for s in filenames if 'ROW_'+np.str(chunks_final[i])+'_'+ np.str(chunks_final[i+1]) in s]
        
        for j in range(len(matching_master)):
            child_file = matching_master[j][0:-1]
            
            print(cdpzn+'/'+child_file)
            
            data = re.findall(r"[-+]?\d*\.\d+|\d+", child_file)
   
            MASTER_indx = int(data[0])
            CHILD_indx = int(data[1])
    
            ROW1 = int(data[2])
            ROW2 = int(data[3])
    
            COL1 = int(data[4])
            COL2 = int(data[5])
    
            child_nc = Dataset(cdpzn+'/'+child_file, 'r')
    
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
            tmp[:,:] = data_child_z
            tmp = np.where(tmp==0.0, 0, 0.5) # create a mask for valid data (~land)
            tmp[:,[0,-1]] = 1.0 # add border of the tile
            tmp[[0,-1],:] = 1.0 # idem
            tile_map[ROW1:ROW2,COL1:COL2] = tmp

        # DEBUG
        v = temp_store_trend
        arr_name = ['pval', 'zval', 'sn', 'len']
        for ii,namei in zip(range(4), arr_name):
            print(f'{namei} NaN/Min/Max:', f'{100*np.isnan(v[:,:,ii]).sum()/v[:,:,ii].size:.1f}%', np.nanmin(v[:,:,ii]), np.nanmax(v[:,:,ii]))

    nc_output.variables['chunk_scores_p_val'][:] = temp_store_trend[:,:,0]
    nc_output.variables['chunk_scores_z_val'][:] = temp_store_trend[:,:,1]
    nc_output.variables['chunk_scores_Sn_val'][:] = temp_store_trend[:,:,2]
    nc_output.variables['chunk_scores_length'][:] = temp_store_trend[:,:,3]
            
    nc_output.close() 

    print('Plot debug tile map...')
    plt.imshow(tile_map)
    plt.savefig('tile_map.png')
    

def plot_trends(input_path, output_path, file_trend_name, xlim1, xlim2, ylim1, ylim2,
                plot_name, plot_choice, scale_tendency):
    
    cdpzn = input_path

    print('*** PLOT TRENDS')
    
    nc_output = Dataset(cdpzn+'/'+file_trend_name,'r')

    trends = {}
    trends['pval'] = nc_output.variables['chunk_scores_p_val'][:]
    trends['zval'] = nc_output.variables['chunk_scores_z_val'][:]
    trends['len'] = nc_output.variables['chunk_scores_length'][:]
    trends['sn'] = nc_output.variables['chunk_scores_Sn_val'][:]
    nc_output.close()
    
        
    work_dir_msg = './input_ancillary/'
    # Reading lat from MSG disk
    latmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
    lat_MSG = latmsg['LAT'][:]
    lat_MSG = np.array(lat_MSG,dtype='float')
    lat_MSG[lat_MSG==910000] = np.nan
    lat_MSG = lat_MSG*0.0001  

    lonmsg = h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
    lon_MSG = lonmsg['LON'][:]
    lon_MSG = np.array(lon_MSG,dtype='float')
    lon_MSG[lon_MSG==910000] = np.nan    
    lon_MSG = lon_MSG*0.0001
    
    
    hangle_view_zen = h5py.File(work_dir_msg+'GEO_L1B-ANGLES-MSG2_2012-06-15T13-00-00_V1-00.hdf5','r')
    view_zenith = hangle_view_zen['View_Zenith'][:]
    view_zenith = np.array(view_zenith,dtype='float')
    view_zenith[view_zenith==65535] = np.NaN
    view_zenith = view_zenith*0.01
    
    lon_MSG[np.isnan(lon_MSG)] = 91
    lat_MSG[np.isnan(lat_MSG)] = 91

 
    trends['sn'][trends['sn']==999] = np.NaN
    trends['sn'][trends['pval']>0.05] = np.NaN

    trends['zval'][trends['zval']==999] = np.NaN
    trends['zval'][trends['pval']>0.05] = np.NaN

    
    proj = ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831, globe=None)
    #fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection':proj})
    fig, ax = plt.subplots(subplot_kw={'projection':proj})

    zone_str = '_'.join([str(i) for i in [xlim1,xlim2,ylim1,ylim2]])

    if plot_choice=='sn':
        trends[plot_choice][view_zenith>75] = np.NaN
        dx = xr.DataArray(trends[plot_choice][xlim1:xlim2,ylim1:ylim2]*scale_tendency, dims = ('y','x'))
        '''The value of scale tendency depends on the product, daily or high frequency, for daily, the value is 365 '''
    else:
        dx = xr.DataArray(trends[plot_choice][xlim1:xlim2,ylim1:ylim2], dims = ('y','x'))
    plot_save = plot_choice+'_'+zone_str+'.png'
    
    dx.coords['lat'] = (('y', 'x'), lat_MSG[xlim1:xlim2,ylim1:ylim2])
    dx.coords['lon'] = (('y', 'x'), lon_MSG[xlim1:xlim2,ylim1:ylim2])

    if plot_choice=='sn':
        #imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), subplot_kws={'projection': proj}, vmin=-1E-5, vmax=1E-5, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), vmin=-1E-5, vmax=1E-5, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)


    if plot_choice=='zval':
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), subplot_kws={'projection': proj}, vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='pval':
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), subplot_kws={'projection': proj}, vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)

    if plot_choice=='len':
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), subplot_kws={'projection': proj}, vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        #imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)
        imm = dx.plot(x='lon', y='lat', transform=ccrs.PlateCarree(), cmap='RdBu_r', add_colorbar=False, extend='neither', ax=ax)


    cbar = plt.colorbar(imm,ax=ax,orientation='horizontal',shrink=.8,pad=0.05,aspect=10)
    cbar.ax.tick_params(labelsize=14,rotation=90)

    ax.coastlines()
    plt.axis('off')
    ax.set_title(plot_name, fontsize=20)
    
    plot_string_save = output_path + '/' + plot_save
    plot_string_save = os.path.normpath(plot_string_save)
    
    #fig.savefig(plot_string_save, dpi=300)
    fig.savefig(plot_string_save, dpi=100)
    plt.close()   

    print('*** IMAGE SAVED >', plot_string_save)





    










   
