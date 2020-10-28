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
from scipy import signal

def merge_tiles(root_cci_year_first,write_file_cci_first,root_cci_year_next,write_file_cci_next,xlim1,xlim2,ylim1,ylim2):
    
    plot_tag='true'
    cdpzn_2004=root_cci_year_first
    cdpzn_2018=root_cci_year_next
    
    '''first CCI read '''
    nc_output=Dataset(cdpzn_2004+'/'+write_file_cci_first,'w')
    xdim=nc_output.createDimension('X_dim',3712)
    ydim=nc_output.createDimension('Y_dim',3712)
    Vardim=nc_output.createDimension('Var',4)
    
    output_data=nc_output.createVariable('chunk_scores_bb',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    output_data=nc_output.createVariable('chunk_scores_needle',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    
    temp_store_trend=np.empty([3712,3712,2])
    temp_store_trend[:]=np.NaN
    
    chunks_main=np.arange(xlim1,xlim2,50)
    chunks_final=np.append(chunks_main,[xlim2],axis=0)
    
    
    
    fp=open(root_cci_year_first+'/'+'fileslist.txt')
    filenames=fp.readlines()
    for i in range(len(chunks_final)-1):
        print (chunks_final[i],chunks_final[i+1])
            
        matching_master = [s for s in filenames if 'ROW_'+np.str(chunks_final[i])+'_'+ np.str(chunks_final[i+1]) in s]
        
        for j in range(len(matching_master)):
            child_file=matching_master[j][0:-1]
            
            print(cdpzn_2004+'/'+child_file)
            
            data=re.findall(r"[-+]?\d*\.\d+|\d+", child_file)
    
            MASTER_indx=np.array(data[0],dtype='i')
            CHILD_indx=np.array(data[1],dtype='i')
    
            ROW1=np.array(data[2],dtype='i')
            ROW2=np.array(data[3],dtype='i')
    
            COL1=np.array(data[4],dtype='i')
            COL2=np.array(data[5],dtype='i')
    
            child_nc=Dataset(cdpzn_2004+'/'+child_file,'r')
    
            data_child_bb=child_nc.variables['scores_bb_cci'][:].astype('f')
            data_child_needle=child_nc.variables['scores_needle_cci'][:].astype('f')
    
    
            child_nc.close()
            
            
            temp_store_trend[ROW1:ROW2,COL1:COL2,0]=data_child_bb
            temp_store_trend[ROW1:ROW2,COL1:COL2,1]=data_child_needle
            
    nc_output.variables['chunk_scores_bb'][:]=temp_store_trend[:,:,0]
    nc_output.variables['chunk_scores_needle'][:]=temp_store_trend[:,:,1]
            
    nc_output.close() 
    
    
    '''second CCI Read '''
    nc_output=Dataset(cdpzn_2018+'/'+write_file_cci_next,'w')
    xdim=nc_output.createDimension('X_dim',3712)
    ydim=nc_output.createDimension('Y_dim',3712)
    Vardim=nc_output.createDimension('Var',4)
    
    output_data=nc_output.createVariable('chunk_scores_bb',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    output_data=nc_output.createVariable('chunk_scores_needle',np.float,('X_dim','Y_dim'),zlib=True,least_significant_digit=3)
    
    temp_store_trend=np.empty([3712,3712,2])
    temp_store_trend[:]=np.NaN
    
    chunks_main=np.arange(xlim1,xlim2,50)
    chunks_final=np.append(chunks_main,[xlim2],axis=0)
    
    
    
    fp=open(root_cci_year_next+'/'+'fileslist.txt')
    filenames=fp.readlines()
    for i in range(len(chunks_final)-1):
        print (chunks_final[i],chunks_final[i+1])
            
        matching_master = [s for s in filenames if 'ROW_'+np.str(chunks_final[i])+'_'+ np.str(chunks_final[i+1]) in s]
        
        for j in range(len(matching_master)):
            child_file=matching_master[j][0:-1]
            
            print(cdpzn_2004+'/'+child_file)
            
            data=re.findall(r"[-+]?\d*\.\d+|\d+", child_file)
    
            MASTER_indx=np.array(data[0],dtype='i')
            CHILD_indx=np.array(data[1],dtype='i')
    
            ROW1=np.array(data[2],dtype='i')
            ROW2=np.array(data[3],dtype='i')
    
            COL1=np.array(data[4],dtype='i')
            COL2=np.array(data[5],dtype='i')
    
            child_nc=Dataset(cdpzn_2004+'/'+child_file,'r')
    
            data_child_bb=child_nc.variables['scores_bb_cci'][:].astype('f')
            data_child_needle=child_nc.variables['scores_needle_cci'][:].astype('f')
    
    
            child_nc.close()
            
            
            temp_store_trend[ROW1:ROW2,COL1:COL2,0]=data_child_bb
            temp_store_trend[ROW1:ROW2,COL1:COL2,1]=data_child_needle
            
    nc_output.variables['chunk_scores_bb'][:]=temp_store_trend[:,:,0]
    nc_output.variables['chunk_scores_needle'][:]=temp_store_trend[:,:,1]
            
    nc_output.close()    
    
    if plot_tag=='true':
        
        nc_output=Dataset(cdpzn_2004+'/'+write_file_cci_first,'r')
        trend_bb_2004=nc_output.variables['chunk_scores_bb'][:]
        trend_needle_2004=nc_output.variables['chunk_scores_needle'][:]
        nc_output.close()
        trend_bb_2004[trend_bb_2004==0]=np.NaN
        trend_needle_2004[trend_needle_2004==0]=np.NaN
    
        nc_output=Dataset(cdpzn_2018+'/'+write_file_cci_next,'r')
        trend_bb_2018=nc_output.variables['chunk_scores_bb'][:]
        trend_needle_2018=nc_output.variables['chunk_scores_needle'][:]
        nc_output.close()
        trend_bb_2018[trend_bb_2018==0]=np.NaN
        trend_needle_2018[trend_needle_2018==0]=np.NaN
    
        diff_needle=trend_needle_2004-trend_needle_2018
        diff_bb=trend_bb_2004-trend_bb_2018
            
        work_dir_msg='./input_ancillary/'
            # Reading lat from MSG disk
        latmsg=h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LAT_MSG-Disk_4bytesPrecision','r')
        lat_MSG=latmsg['LAT'][:]
        lat_MSG=np.array(lat_MSG,dtype='float')
        lat_MSG[lat_MSG==910000]=np.nan
        lat_MSG=lat_MSG*0.0001  
    
        lonmsg=h5py.File(work_dir_msg+'HDF5_LSASAF_MSG_LON_MSG-Disk_4bytesPrecision','r')
        lon_MSG=lonmsg['LON'][:]
        lon_MSG=np.array(lon_MSG,dtype='float')
        lon_MSG[lon_MSG==910000]=np.nan    
        lon_MSG=lon_MSG*0.0001
        
        lon_MSG[np.isnan(lon_MSG)]=91
        lat_MSG[np.isnan(lat_MSG)]=91
        
     
    #    proj=ccrs.PlateCarree(central_longitude=0.0, globe=None)
        proj=ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831, globe=None)
        fig, axes = plt.subplots(figsize=(12, 12),subplot_kw={'projection':proj})
    
        dx = xr.DataArray(diff_needle[0:800,1500:3500], dims = ('y','x'))
        dx.coords['lat']= (('y', 'x'), lat_MSG[0:800,1500:3500])
        dx.coords['lon']= (('y', 'x'), lon_MSG[0:800,1500:3500])
        ax=axes
    
            
        imm=dx.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), subplot_kws={'projection': proj},vmin=-10,vmax=10,cmap='jet',add_colorbar=False, extend='neither',ax=ax)
    
        cbar=plt.colorbar(imm,ax=ax,orientation='horizontal',shrink=.8,pad=0.05,aspect=10)
        cbar.ax.tick_params(labelsize=14,rotation=90)
    
        ax.coastlines()
        plt.axis('off')
        ax.set_title('ESA CCI needle leaves \n (2004-2018)',fontsize=20)
        
        plot_string_save='./output_plots/ccimerger/scores_needle_leaves_final_2004_2018.png'
        
        fig.savefig(plot_string_save,dpi=300)
        plt.close()   
    
    
    
    '''plotting histogram for the % difference of CCI land covers  '''
    diff_mat=np.reshape(diff_bb[0:800,1500:3500],800*2000)
    diff_mat[diff_mat==0]=np.NaN
    diff_mat=diff_mat
    
    fig=plt.figure(figsize=(12,12))
    binrange=np.arange(-100,100,0.5)
    center = (binrange[:-1] + binrange[1:]) / 2

    imp = signal.unit_impulse(200, 'mid')
    plt.plot(np.arange(-100, 100), imp*4500,'black')
    maskup=np.where(binrange[:-1]>=30)
    maskdown=np.where(binrange[:-1]<=-30)
    indx_both=np.where(np.logical_or(binrange[:-1]<=-30,binrange[:-1]>=30))
    N,X = np.histogram(diff_mat, bins = binrange)
    estimate_change_positive=np.nanmean(N[maskup]*3.0*3.0)
    estimate_change_negative=np.nanmean(N[maskdown]*3.0*3.0)
    mean_change_negative_positive=np.nanmean(N[indx_both]*3.0*3.0)
    
    corstattext=' mean positive change = %d km$^{2}$\n mean negative change = %d km$^{2}$  \n mean overall change = %d km$^{2}$'%(estimate_change_negative,estimate_change_positive,mean_change_negative_positive)
    plt.text(-80,5300,corstattext, color='k', fontsize=12, bbox={'fc':'white','ec':'white', 'alpha':0.9, 'pad':10})
    
    plt.step(X[:-1],N*3*3,'k',linewidth=1.0)
    plt.fill_between(X[:-1][maskup],N[maskup]*3*3, 0,
                     facecolor="red", # The fill color
                     color='red',       # The outline color
                     alpha=1)  #plt.hist(diff_mat[mask],bins=binrange,histtype='bar', color='red', lw=0)
    plt.fill_between(X[:-1][maskdown],N[maskdown]*3*3, 0,
                     facecolor="blue", # The fill color
                     color='blue',       # The outline color
                     alpha=1)  #plt.hist(diff_mat[mask],bins=binrange,histtype='bar', color='red', lw=0)
    
    plt.ylim([0,6000])  
    #plt.xlim([-10,10])
    plt.ylabel('km$^{2}$',fontsize=20)
    plt.xlabel('% change',fontsize=20)     
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)      
    plt.title('ESA CCI broad leaves\n (2004-2018)',fontsize=20)           
    plt.savefig('./output_plots/ccimerger/bins_bb_esa_cci_2004_2018.png',dpi=300)
    plt.close()
        
    ''' Needle Class'''
    diff_mat=np.reshape(diff_needle[xlim1:xlim2,ylim1:ylim2],(xlim2-xlim1)*(ylim2-ylim1))
    diff_mat[diff_mat==0]=np.NaN
    diff_mat=diff_mat
    
    fig=plt.figure(figsize=(12,12))
    binrange=np.arange(-100,100,0.5)
    center = (binrange[:-1] + binrange[1:]) / 2

    imp = signal.unit_impulse(200, 'mid')
    plt.plot(np.arange(-100, 100), imp*6500,'black')
    maskup=np.where(binrange[:-1]>=30)
    maskdown=np.where(binrange[:-1]<=-30)
    indx_both=np.where(np.logical_or(binrange[:-1]<=-30,binrange[:-1]>=30))
    N,X = np.histogram(diff_mat, bins = binrange)
    estimate_change_positive=np.nanmean(N[maskup]*3.0*3.0)
    estimate_change_negative=np.nanmean(N[maskdown]*3.0*3.0)
    mean_change_negative_positive=np.nanmean(N[indx_both]*3.0*3.0)
    
    corstattext=' mean positive change = %d km$^{2}$\n mean negative change = %d km$^{2}$  \n mean overall change = %d km$^{2}$'%(estimate_change_negative,estimate_change_positive,mean_change_negative_positive)
    plt.text(-80,6200,corstattext, color='k', fontsize=12, bbox={'fc':'white','ec':'white', 'alpha':0.9, 'pad':10})
    
    plt.step(X[:-1],N*3*3,'k',linewidth=1.0)
    plt.fill_between(X[:-1][maskup],N[maskup]*3*3, 0,
                     facecolor="red", # The fill color
                     color='red',       # The outline color
                     alpha=1)  #plt.hist(diff_mat[mask],bins=binrange,histtype='bar', color='red', lw=0)
    plt.fill_between(X[:-1][maskdown],N[maskdown]*3*3, 0,
                     facecolor="blue", # The fill color
                     color='blue',       # The outline color
                     alpha=1)  #plt.hist(diff_mat[mask],bins=binrange,histtype='bar', color='red', lw=0)
    
    plt.ylim([0,7000])  
    plt.ylabel('km$^{2}$',fontsize=20)
    plt.xlabel('% change',fontsize=20)     
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)      
    plt.title('ESA CCI needle leaves\n (2004-2018)',fontsize=20)           
    plt.savefig('./output_plots/ccimerger/bins_needle_esa_cci_2004_2018.png',dpi=300)
    plt.close()








   
