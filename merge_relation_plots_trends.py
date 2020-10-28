#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:00:01 2019
This part of the code will create
trends plots for each product,
LAI, LST, EVAPO, ALBEDO pairing each dependency

library seaborn 

@author: moparthys
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import xarray as xr
import matplotlib as mpl
import h5py
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib import gridspec
from scipy import stats


'''Merged albedo MDAL NRT, ICARE, MDAL_R '''

savefile='./output_plots/mergertrends/merger_trends.png'

cdpzn_2004='/cnrm/vegeo/SAT/CODES_suman/From_Suman/clean_trends/output_cci_chunks/process_weights_two_2004'
#cdpzn_2005='/cnrm/vegeo/SAT/CODES/From_Suman/prepare_seviri_projection/process_weights_two_2005'
cdpzn_2018='/cnrm/vegeo/SAT/CODES_suman/From_Suman/clean_trends/output_cci_chunks/process_weights_two_2018'


nc_output=Dataset(cdpzn_2004+'/Merger_cci_scores_2004_new.nc','r')
trend_bb_2004=nc_output.variables['chunk_scores_bb'][:]
trend_needle_2004=nc_output.variables['chunk_scores_needle'][:]
nc_output.close()

trend_bb_2004[trend_bb_2004==0]=np.NaN
trend_needle_2004[trend_needle_2004==0]=np.NaN

nc_output=Dataset(cdpzn_2018+'/Merger_cci_scores_2018.nc','r')
trend_bb_2018=nc_output.variables['chunk_scores_bb'][:]
trend_needle_2018=nc_output.variables['chunk_scores_needle'][:]
nc_output.close()

trend_bb_2018[trend_bb_2018==0]=np.NaN
trend_needle_2018[trend_needle_2018==0]=np.NaN

diff_needle=trend_needle_2004-trend_needle_2018
diff_bb=trend_bb_2004-trend_bb_2018


'''Load LAT and LON here '''
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


Dataset_trends=np.empty([3712*3712,4])


'''Albedo trends '''


'''start with 50% change and slowly decrease it till you get atleast 1000 points for comparison '''

nc_output=Dataset('./trends_together/Merger_TRND_ALBEDO_Final.nc','r')

trend_pval=nc_output.variables['chunk_scores_p_val'][:]
trend_zval=nc_output.variables['chunk_scores_z_val'][:]
trend_len=nc_output.variables['chunk_scores_length'][:]
trend_sn=nc_output.variables['chunk_scores_Sn_val'][:]
nc_output.close()


trend=trend_sn.copy()
trend[trend==999]=np.NaN
trend[trend_pval>0.05]=np.NaN
#trend_copy=trend[440:475,1825:1850].copy() # Landes Forest
trend_copy=trend.copy() # Landes Forest
#trend_copy[indx_cci_plot]=np.NaN

#Dataset_trends=np.empty([trend_copy.shape[0]*trend_copy.shape[1],4])

trend_copy=np.reshape(trend_copy,[trend_copy.shape[0]*trend_copy.shape[1]])
Dataset_trends[:,0]=trend_copy*365



'''LAI Daily trends'''
nc_output=Dataset('./trends_together/Merger_TRND_LAI_Final.nc','r')

trend_pval=nc_output.variables['chunk_scores_p_val'][:]
trend_zval=nc_output.variables['chunk_scores_z_val'][:]
trend_len=nc_output.variables['chunk_scores_length'][:]
trend_sn=nc_output.variables['chunk_scores_Sn_val'][:]
nc_output.close()

trend=trend_sn.copy()
trend[trend==999]=np.NaN
trend[trend_pval>0.05]=np.NaN
#trend_copy=trend[440:475,1825:1850].copy() # Landes Forest
trend_copy=trend.copy() # Landes Forest
#trend_copy[indx_cci_plot]=np.NaN
trend_copy=np.reshape(trend_copy,[trend_copy.shape[0]*trend_copy.shape[1]])

Dataset_trends[:,1]=trend_copy*365



'''LST 13UTC trends'''
nc_output=Dataset('./trends_together/Merger_TRND_LST_Final.nc','r')
'''Replace this file by LST; hourly vise computed files '''

trend_pval=nc_output.variables['chunk_scores_p_val'][:]
trend_zval=nc_output.variables['chunk_scores_z_val'][:]
trend_len=nc_output.variables['chunk_scores_length'][:]
trend_sn=nc_output.variables['chunk_scores_Sn_val'][:]
nc_output.close()

trend=trend_sn.copy()
trend[trend==999]=np.NaN
trend[trend_pval>0.05]=np.NaN
#trend_copy=trend[440:475,1825:1850].copy() # Landes Forest
trend_copy=trend.copy() # Landes Forest
#trend_copy[indx_cci_plot]=np.NaN
trend_copy=np.reshape(trend_copy,[trend_copy.shape[0]*trend_copy.shape[1]])


Dataset_trends[:,2]=trend_copy*365



'''Evapotranspiration trends'''

nc_output=Dataset('./trends_together/Merger_TRND_EVAPO_Final.nc','r')

trend_pval=nc_output.variables['chunk_scores_p_val'][:]
trend_zval=nc_output.variables['chunk_scores_z_val'][:]
trend_len=nc_output.variables['chunk_scores_length'][:]
trend_sn=nc_output.variables['chunk_scores_Sn_val'][:]
nc_output.close()


trend=trend_sn.copy()
trend[trend==999]=np.NaN
trend[trend_pval>0.05]=np.NaN
#trend_copy=trend[440:475,1825:1850].copy() # Landes Forest
trend_copy=trend.copy() # Landes Forest
#trend_copy[indx_cci_plot]=np.NaN
trend_copy=np.reshape(trend_copy,[trend_copy.shape[0]*trend_copy.shape[1]])

Dataset_trends[:,3]=trend_copy*365



'''Relation plots here '''

Dataset_trends_cci_based=Dataset_trends.copy()
#diff_cci=diff_bb.copy()[440:475,1825:1850]
diff_cci=diff_needle.copy()

#diff_cci=np.absolute(diff_bb)
diff_cci=np.reshape(diff_cci,[3712*3712])
#diff_cci=np.reshape(diff_cci,[35*25])

indx_cci_loss=np.where(diff_cci>=30)
trend_cci_2004=np.reshape(trend_needle_2004,[3712*3712])
trend_cci_2018=np.reshape(trend_needle_2018,[3712*3712])

trend_cci_concate_loss = np.array([trend_cci_2004[indx_cci_loss], trend_cci_2018[indx_cci_loss]])
average_cci_loss=np.nanmean(trend_cci_concate_loss, axis=0)

indx_cci_gain=np.where(diff_cci<=-30)
trend_cci_concate_gain = np.array([trend_cci_2004[indx_cci_gain], trend_cci_2018[indx_cci_gain]])
average_cci_gain=np.nanmean(trend_cci_concate_gain, axis=0)



#Dataset_trends_cci_based[indx_cci_plot[0],:]=np.NaN

hlw=h5py.File('./input_ancillary/HDF5_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_201610171300','r')
#lwmsk=hlw['LWMASK'][:][440:475,1825:1850]
lwmsk=hlw['LWMASK'][:]

lwmsk=np.reshape(lwmsk,[3712*3712])
#lwmsk=np.reshape(lwmsk,[35*25])

find_ocean=np.where(lwmsk==0)
find_land=np.where(lwmsk==1)

Dataset_trends_cci_based[find_ocean[0],:]=np.NaN


hangle_view_zen=h5py.File('./input_ancillary/GEO_L1B-ANGLES-MSG2_2012-06-15T13-00-00_V1-00.hdf5','r')
#view_zenith=hangle_view_zen['View_Zenith'][:][440:475,1825:1850]
view_zenith=hangle_view_zen['View_Zenith'][:]
view_zenith=np.array(view_zenith,dtype='float')
view_zenith[view_zenith==65535]=np.NaN
view_zenith=view_zenith*0.01

view_zenith=np.reshape(view_zenith,[3712*3712])
#view_zenith=np.reshape(view_zenith,[35*25])

Dataset_trends_cci_based[view_zenith>75,:]=np.NaN


lat_MSG=np.reshape(lat_MSG,[3712*3712])
#lon_MSG=np.reshape(lon_MSG,[3712*3712])

#indx_lat_lon=np.where(np.logical_and(np.logical_or(lat_MSG<43,lat_MSG>45),np.logical_or(lon_MSG<-1,lon_MSG>1.3)))
#indx_lat=np.where(np.logical_or(lat_MSG<42.5,lat_MSG>51))
#indx_lon=np.where(np.logical_or(lon_MSG<-4.5,lon_MSG>8))

#Dataset_trends_cci_based[indx_lat,:]=np.NaN
#Dataset_trends_cci_based[indx_lon,:]=np.NaN

#Dataset_trends_cci_based[lat_MSG<=60,:]=np.NaN
'''high light this when points or pixels needed lat > 60 '''

Dataset_trends_cci_based[lat_MSG>60,:]=np.NaN 
'''high light this when points or pixels needed lat <=60 '''


df=pd.DataFrame(Dataset_trends_cci_based,columns=['AL-BB-DH','LAI','LST','Evapotranspiration'])

fig = plt.figure(figsize=(32, 24) )
gs = gridspec.GridSpec(4, 4, width_ratios=[6, 6, 6, 6])
positions_subplot=np.array([0,4,8,12])
for i in range(4):
    for j in range(4):
        '''plot relation between 4 variables and close the subplot; use different colors for cci class '''
        
        ax = plt.subplot(gs[i,j])
        x=df.values[:,i]
        y=df.values[:,j]
        labelx=df.columns[i]
        labely=df.columns[j]

#        ax.plot(x,y,'.k',markersize=1.5)
#        ax.plot(x[indx_cci_gain],y[indx_cci_gain],'s',markersize=8.0)
#        ax.plot(x[indx_cci_loss],y[indx_cci_loss],'o',markersize=8.0)

#        c1=ax.scatter(x[indx_cci_gain],y[indx_cci_gain],marker='s', edgecolors='b',c=average_cci_gain, alpha=1, s=20, cmap='jet',vmin=30, vmax=100)
#        c2=ax.scatter(x[indx_cci_loss],y[indx_cci_loss],marker='o', edgecolors='r',c=average_cci_loss, alpha=1, s=20, cmap='jet',vmin=30, vmax=100)
        c1=ax.scatter(x[indx_cci_gain],y[indx_cci_gain], marker='s', edgecolors='cyan', s=30, c=average_cci_gain, cmap='jet', vmin=30, vmax=100)
#        c1.set_facecolor('none')

#        closs= plt.cm.jet(average_cci_loss)
        c2=ax.scatter(x[indx_cci_loss],y[indx_cci_loss], marker='o', s=30,  c=average_cci_loss, cmap='jet', vmin=30, vmax=100)
        c2.set_facecolor('none')



#        c1=ax.plot(x[indx_cci_gain],y[indx_cci_gain],marker='s', markeredgecolor=average_cci_gain,markerfacecolor='cyan', alpha=1, markersize=10, cmap='jet',vmin=30, vmax=100)
#        c2=ax.plot(x[indx_cci_loss],y[indx_cci_loss],marker='o', markeredgecolor=average_cci_loss,markerfacecolor='None', alpha=1, markersize=10, cmap='jet',vmin=30, vmax=100)

#        ax.colorbar(c1)
        
        ax.set_xlabel(labelx,fontsize=14)
        ax.set_ylabel(labely,fontsize=14) 
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.axhline(0, color='k')
        ax.axvline(0, color='k')        
 
        if df.columns[i]=='AL-BB-DH':
            ax.set_xlim([-0.005,0.005])

            if df.columns[j]=='AL-BB-DH':
                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg2=np.arange(-0.005,0.005+0.0005,0.0005)

            if df.columns[j]=='LST':
                ax.set_ylim([-1,1])
                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg2=np.linspace(-1,1,len(vals_reg1))

            if df.columns[j]=='LAI':
                ax.set_ylim([-0.1,0.1])
                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg2=np.linspace(-0.1,0.1,len(vals_reg1))

            if df.columns[j]=='Evapotranspiration':
#                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
#                vals_reg2=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg2=np.linspace(np.nanmin(y),np.nanmax(y)+0.0005,len(vals_reg1))


        if df.columns[i]=='LST':
            ax.set_xlim([-1,1])

            if df.columns[j]=='AL-BB-DH':
                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-1,1+0.1,0.1)
                vals_reg2=np.linspace(-0.005,0.005,len(vals_reg1))

            if df.columns[j]=='LST':
                ax.set_ylim([-1,1])
                vals_reg1=np.arange(-1,1.1,0.1)
                vals_reg2=np.arange(-1,1.1,0.1)

            if df.columns[j]=='LAI':
                ax.set_ylim([-0.1,0.1])
                vals_reg1=np.arange(-1,1.1,0.1)
                vals_reg2=np.linspace(-0.1,0.1,len(vals_reg1))

            if df.columns[j]=='Evapotranspiration':
#                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-1,1.1,0.1)
#                vals_reg2=np.linspace(-0.005,0.005,len(vals_reg1))
                vals_reg2=np.linspace(np.nanmin(y),np.nanmax(y)+0.0005,len(vals_reg1))


        if df.columns[i]=='LAI':
            ax.set_xlim([-0.1,0.1])

            if df.columns[j]=='AL-BB-DH':
                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-0.1,0.1+0.01,0.01)
                vals_reg2=np.linspace(-0.005,0.005,len(vals_reg1))

            if df.columns[j]=='LST':
                ax.set_ylim([-1,1])
                vals_reg1=np.arange(-0.1,0.1+0.01,0.01)
                vals_reg2=np.linspace(-1,1,len(vals_reg1))

            if df.columns[j]=='LAI':
                ax.set_ylim([-0.1,0.1])
                vals_reg1=np.arange(-0.1,0.1+0.01,0.01)
                vals_reg2=np.arange(-0.1,0.1+0.01,0.01)

            if df.columns[j]=='Evapotranspiration':
#                ax.set_ylim([-0.005,0.005])
                vals_reg1=np.arange(-0.1,0.1+0.01,0.01)
#                vals_reg2=np.linspace(-0.005,0.005,len(vals_reg1))
                vals_reg2=np.linspace(np.nanmin(y),np.nanmax(y)+0.0005,len(vals_reg1))


        if df.columns[i]=='Evapotranspiration':
#            ax.set_xlim([-0.005,0.005])

            if df.columns[j]=='AL-BB-DH':
                ax.set_ylim([-0.005,0.005])
#                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg1=np.arange(np.nanmin(x),np.nanmax(x)+0.0005,0.0005)
                
#                vals_reg2=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg2=np.linspace(-0.005,0.005+0.0005,len(vals_reg1))

            if df.columns[j]=='LST':
                ax.set_ylim([-1,1])
#                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg1=np.arange(np.nanmin(x),np.nanmax(x)+0.0005,0.0005)

                vals_reg2=np.linspace(-1,1,len(vals_reg1))

            if df.columns[j]=='LAI':
                ax.set_ylim([-0.1,0.1])
#                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
                vals_reg1=np.arange(np.nanmin(x),np.nanmax(x)+0.0005,0.0005)

                vals_reg2=np.linspace(-0.1,0.1,len(vals_reg1))

            if df.columns[j]=='Evapotranspiration':
#                ax.set_ylim([-0.005,0.005])
#                vals_reg1=np.arange(-0.005,0.005+0.0005,0.0005)
#                vals_reg2=np.arange(-0.005,0.005+0.0005,0.0005)

                vals_reg1=np.arange(np.nanmin(y),np.nanmax(y)+0.0005,0.0005)
                vals_reg2=np.arange(np.nanmin(y),np.nanmax(y)+0.0005,0.0005)


        nanindx=np.where(np.logical_and(np.isnan(x[indx_cci_gain])!=1,np.isnan(y[indx_cci_gain])!=1))
        xxgain=x[indx_cci_gain][nanindx]
        yygain=y[indx_cci_gain][nanindx]
        
        slope_CI, intercept_CI, r_value_CI, p_value_CI, std_err_CI = stats.linregress(xxgain, yygain)

        ax.plot(vals_reg1, slope_CI*vals_reg1+intercept_CI,'b-',linewidth=2.0)

        nanindx=np.where(np.logical_and(np.isnan(x[indx_cci_loss])!=1,np.isnan(y[indx_cci_loss])!=1))
        xxloss=x[indx_cci_loss][nanindx]
        yyloss=y[indx_cci_loss][nanindx]

        slope_CI, intercept_CI, r_value_CI, p_value_CI, std_err_CI = stats.linregress(xxloss, yyloss)
        ax.plot(vals_reg1, slope_CI*vals_reg1+intercept_CI,'r-',linewidth=2.0)

        ax.plot(vals_reg1,vals_reg2,'g--',linewidth=2.0) # Draw a reference line from (0,0) to (1,1)


        ax.tick_params(axis='both', which='minor', labelsize=8)
        
#        sm=scatter_matrix(df, alpha = 0.2, figsize = (12, 12), diagonal='kde',color="black")
#        [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
#        [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#        [s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]
plt.savefig(savefile,dpi=300)
#plt.savefig('/cnrm/vegeo/SAT/CODES/From_Suman/Trends_and_Time_series/trend_relation_merger_needle_leaves_cci_10_scatter_no_lat',dpi=300)

plt.close()        
