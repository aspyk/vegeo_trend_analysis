import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py
import sys
import generic
import tools


class Main:
    def __init__(self):
        self.icol = 0
        self.col = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        self.imshow = []
        self.imshow_clim = [999999, -999999]

    def plot_grid(self, f, slat, slon):
        
        with h5py.File(f, mode='r') as h:
            lon = h['lon'][slon]
            lat = h['lat'][slat]

            print(lat, lon)

            dlon = np.diff(h['lon'][:10]).mean()/2.
            dlat = np.diff(h['lat'][:10]).mean()/2.
            #print("dlat,dlon=", dlat,dlon)
        
        xlon = np.linspace(lon[0]-dlon, lon[-1]+dlon, len(lon)+1)
        ylat = np.linspace(lat[0]-dlat, lat[-1]+dlat, len(lat)+1)

        x, y = np.meshgrid(xlon, ylat)
        
        self.ax.plot(x, y, c=self.col[self.icol], lw=1)
        self.ax.plot(np.transpose(x), np.transpose(y), c=self.col[self.icol]) 
        self.icol += 1
        
    def save_avhrr_ref_corner(self, f, out_csv):
        """
        Save as a new csv file the lat/lon coordinate of the top-left corner of
        the AVHRR pixel containing the initial coor of the landval sites.
        It will then be used as reference to extract fine grid.
        """
        

        with h5py.File(f, mode='r') as h:
            dlon = np.diff(h['lon'][:10]).mean()/2.
            dlat = np.diff(h['lat'][:10]).mean()/2.

            latlon_topleft = np.array([[h['lat'][ilat]-dlat, h['lon'][ilon]-dlon] for ilat,ilon in self.coor_avhrr.site_coor[['ilat', 'ilon']].values])

        
        print(latlon_topleft)

        df_new = self.coor_avhrr.site_coor.copy()

        df_new['LATITUDE'] = latlon_topleft.T[0]
        df_new['LONGITUDE'] = latlon_topleft.T[1]
        df_new = df_new.sort_index()
        df_new.to_csv(out_csv,
                      columns=['LATITUDE','LONGITUDE','NAME'],
                      sep=';')


    def plot_scatter(self, lat, lon):
        self.ax.scatter(lon, lat)

    def plot_imshow(self, f, slat, slon, var):

        with h5py.File(f, mode='r') as h:
            lon = h['lon'][slon]
            lat = h['lat'][slat]
            data = h[var][(0, slat, slon)]
            data_err = h[var+'_ERR'][(0, slat, slon)]
            data_age = h['AGE'][(0, slat, slon)]
            data_qflag = h['QFLAG'][(0, slat, slon)]

            dlon = np.diff(h['lon'][:10]).mean()/2.
            dlat = np.diff(h['lat'][:10]).mean()/2.
            #print("dlat,dlon=", dlat,dlon)
        
        print("data=\n", data)
        print("data.mean=\n", data.mean())
        print("data.shape=", data.shape)

        # Update clim
        dmin = np.nanmin(data)
        dmax = np.nanmax(data)
        print("dmin,dmax=", dmin,dmax )
        if self.imshow_clim[0] > dmin:
            self.imshow_clim[0] = dmin
        if self.imshow_clim[1] < dmax:
            self.imshow_clim[1] = dmax

        #extent = [lon[0]-dlon, lon[-1]+dlon, lat[0]-dlat, lat[-1]+dlat]
        extent = [lon[0]-dlon, lon[-1]+dlon, lat[-1]+dlat, lat[0]-dlat] # lat are in decreasing order

        #print('extent=', extent)

        cm = 'jet'
        cm = 'gist_ncar'
        self.imshow.append(self.ax.imshow(data, extent=extent, cmap='gray'))


    def extract_land_mask(self, f1):
        with h5py.File(f1, mode='r') as h:
            lon = h['lon'][:]
            lat = h['lat'][:]
            mask = (h['QFLAG'][:] & 0b11)==1

        with h5py.File('c3s_land_mask.h5', mode='w') as h:
            h.create_dataset('lon', data=lon, compression='gzip')
            h.create_dataset('lat', data=lat, compression='gzip')
            h.create_dataset('mask', data=mask, dtype='bool', compression='gzip')

    def plot_extract_coor(self, csvcoorfile, f1, f2):
        """
        Plot pixel box data around a coordinate
        """

        assert ('AVHRR' in f1) 
        assert ('VGT' in f2) 

        #origin = 'center'
        origin = 'topleft'

        ## Get the input data
        self.coor_avhrr = generic.CoordinatesConverter(csvcoorfile, sensor='AVHRR')
        if origin=='center':
            self.coor_vgt = generic.CoordinatesConverter(csvcoorfile, sensor='VGT')
        elif origin=='topleft':
            out_avhrr_csv_name = 'LANDVAL2_avhrr_topleft.csv'
            self.save_avhrr_ref_corner(f1, out_avhrr_csv_name)
            self.coor_vgt = generic.CoordinatesConverter(out_avhrr_csv_name, sensor='VGT')

        ## Get the indices of the box around a point
        #pt = 'DOM1'
        pt = 'FRENCHMAN_FLAT'

        print(self.coor_avhrr.get_row_by_name(pt))
        print(self.coor_vgt.get_row_by_name(pt))

        #sys.exit()

        slat1, slon1 = self.coor_avhrr.get_box_around(pt, 1)
        if origin=='center':
            slat2, slon2 = self.coor_vgt.get_box_around(pt, 4)
        elif origin=='topleft':
            slat2, slon2 = self.coor_vgt.get_box_from_topleft(pt, 4)

        #print(slat1, slon1)
        #print(slat2, slon2)

        self.fig, self.ax = plt.subplots()
        self.ax.title.set_text(pt)

        ## Plot the data
        var = 'AL_DH_BB'
        #var = 'AGE'
        #var = 'AL_DH_BB_ERR'
        #var = 'QFLAG'
        self.plot_imshow(f1, slat1, slon1, var)
        self.plot_imshow(f2, slat2, slon2, var)
        
        ## Plot the grids
        self.plot_grid(f1, slat1, slon1)
        self.plot_grid(f2, slat2, slon2)

        ## Plot the point
        lat, lon = self.coor_avhrr.get_row_by_name(pt)[['LATITUDE', 'LONGITUDE']].values.T
        self.plot_scatter(lat, lon)
        
        lat, lon = self.coor_vgt.get_row_by_name(pt)[['LATITUDE', 'LONGITUDE']].values.T
        self.plot_scatter(lat, lon)

    def show(self):
        plt.gca().axis('equal')
        
        ## Set colorscale identical for all imshow
        vmin, vmax = self.imshow_clim
        for im in self.imshow:
            im.set_clim(vmin=vmin, vmax=vmax)
        #self.fig.colorbar(im, ax=self.ax) 
        self.fig.colorbar(im, ax=self.ax) 
        
        plt.show()


if __name__=='__main__':

    kwargs = tools.parse_args()

    m = Main()
    #m.plot_grid(**kwargs) # input args:  f1=<path>:f2=<path>
    #m.extract_land_mask(**kwargs) # input args: f1=<path>
    m.plot_extract_coor(**kwargs)
    m.show()


