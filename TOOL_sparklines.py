import numpy as np
from netCDF4 import Dataset
import sys, glob, os
import pathlib
from datetime import datetime
import pandas as pd

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

from io import BytesIO

class Sparkline():
    
    h1 = textwrap.dedent("""\
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" href="mystyle.css">
    </head>
    <body>
    """)
    
    h2 = textwrap.dedent("""\
    </body>
    </html>
    """)
    
    ## Sensor dates
    #NOAA_satellite Start_date End_date
    dic_sen = {}
    dic_sen['NOAA7']  =  ('20-09-1981', '31-12-1984')
    dic_sen['NOAA9']  =  ('20-03-1985', '10-11-1988')
    dic_sen['NOAA11'] =  ('30-11-1988', '20-09-1994')
    dic_sen['NOAA14'] =  ('10-02-1995', '10-03-2001')
    dic_sen['NOAA16'] =  ('20-03-2001', '10-09-2002')
    dic_sen['NOAA17'] =  ('20-09-2002', '31-12-2005')

    dic_sen = {k: [datetime.strptime(i, "%d-%m-%Y").timestamp() for i in v] for k,v in dic_sen.items()}


    def __init__(self, data):
        self.data_group = data
        self.fname_param = []
        pass


    def _add_data_graph(self):
        if mode=='oneline':
            ## Plot sparkline
            self.ax.plot(self.dates, data, c=u'#1f77b4')
            self.ax.fill_between(range(len(data)), data, len(data)*[dmin], alpha=0.1)
 
            ## Add a red dot at the right hand end of the line
            #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    


    def get_graph_image(self, data, mode='wrap', text='', **kwags):
        """
        Returns a HTML image tag containing a base64 encoded sparkline style plot

        Parameters
        ----------
        data: list of dataframe [<timestamp>,<point_name>]
        mode: string, 'oneline' or 'wrap'
        """
    
        #dmin = np.nanmin(data)
        #dmax = np.nanmax(data)

        dmin = self.data_group.gdmin
        dmax = self.data_group.gdmax

        tmin = self.data_group.gtmin
        tmax = self.data_group.gtmax

        if mode=='oneline':

            if self.figsize is None:
                self.figsize = (18, .5)
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, **kwags)

            ## Add sensor time range
            for k,v in self.dic_sen.items():
                #ax.plot(v, mx)
                self.ax.fill_between(v, 2*[dmax], 2*[dmin], alpha=0.1)
                # Plot name
                plt.text(0.5*(v[1]+v[0]), 1.2*dmax, k, horizontalalignment='center', fontsize=8)
            
            ## Add horizontal line and label for max
            self.ax.axhline(dmax, c='k', alpha=0.1)
            # Plot name
            plt.text(tmin, 1.2*dmax, text[:18], horizontalalignment='left', fontsize=8)
            # Plot max
            plt.text(tmax, 1.2*dmax, '{:.2f}'.format(dmax), horizontalalignment='right', fontsize=8)

            # Add vertical lines for years
            #for i in range(int(len(data)/36)):
            #    ax.axvline(i+1, c='k', alpha=0.1)

            ## Add data
            for ts in data:
                self.ax.plot(ts.iloc[:,0].values, ts.iloc[:,1].values)


            self.ax.set_xlim(tmin, tmax)

            ## TODO
            # - plot vline for years
            # - plot limit between sensor
            # - plot several files using dates

        
        elif mode=='wrap':
            
            if self.figsize is None:
                self.figsize = (2, .5)
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, **kwags)

            d = data.reshape((-1,36))
            for y in d:
                self.ax.plot(y, c='b', alpha=1./5.)
            
            ## Add a red dot at the right hand end of the line
            #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    
            #ax.fill_between(range(len(data)), data, len(data)*[np.nanmin(data)], alpha=0.1)
            
            ## Add horizontal line and label for max
            self.ax.axhline(dmax, c='k', alpha=0.1)
            # Plot name
            plt.text(0, 1.2*dmax, text[:18], horizontalalignment='left', fontsize=8)
            # Plot max
            plt.text(36, 1.2*dmax, '{:.2f}'.format(dmax), horizontalalignment='right', fontsize=8)
            self.ax.set_xlim(0,36)
        
        ## Remove all ticks and frame to make sparkline style 
        if 1: 
            for k,v in self.ax.spines.items():
                v.set_visible(False)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
    
    
        #plt.title('Mn/Mx: {:.2f} / {:.2f}'.format(np.nanmin(data), np.nanmax(data)), fontsize=8)
        
    
        img = BytesIO()
        plt.savefig(img, transparent=True, bbox_inches='tight')
        img.seek(0)
        plt.close()
    
        return base64.b64encode(img.read()).decode("UTF-8")    
    
    
   
    
    def run(self, vname, mode, figsize=None):
    
        self.figsize = figsize

        self.data_group.use_var(vname)

        #self.data_group.sort('max')

        ## Param to output in filename
        self.fname_param.append(vname)
        self.fname_param.append(mode)

        ## Fill the output html file
        with open("sparklines_{}.html".format('_'.join(self.fname_param)), "w") as f:
            f.write(self.h1)
            for ip,p in enumerate(self.data_group.point_names):
                if ip%50==0:
                    print(ip)
                # Get the image in base64 format
                img = self.get_graph_image(self.data_group.get_point(p), mode, p)
                f.write('<div><img src="data:image/png;base64,{}"/></div>'.format(img))
            f.write(self.h2)


class TimeSeriesGroup():
    """
    Group time series with common source points.

    Point names ares the keys of the data: time series with common
    key will be process together (plot, merge...)

    TODO
    - point sources could be independant between input files
    """

    def __init__(self, flist):
        self.tslist = []
        for f in flist:
            self.tslist.append(TimeSeries(f))

        self._check()

    def _check(self):
        ## Check available points
        pass
    
    def use_var(self, vname):
        self.var = vname
        for ts in self.tslist:
            ts.use_var(vname)
        self.gtmin = 1e12 
        self.gtmax = -99999999
        for ts in self.tslist:
            if ts.tmin<self.gtmin:
                self.gtmin = ts.tmin
            if ts.tmax>self.gtmax:
                self.gtmax = ts.tmax

        ## Column 0 is the timestamp
        # Timestamp is not the index to keep the order with missing data
        # TODO: Make union of all point names if difference between the input files
        self.point_names = self.tslist[0].df.columns.values.tolist()[1:]

    def get_point(self, p):
        """
        Return a list of dataframe(s) as [date, value] corresponding the point p
        given in argument.
        """
        res = []
        for ts in self.tslist:
            res.append(ts.df[['timestamp',p]])

        ## Update group min and max
        dmin = 99999999
        dmax = -99999999
        for ts in res:
            if np.nanmin(ts[p].values)<dmin:
                dmin = np.nanmin(ts[p].values)
            if np.nanmax(ts[p].values)>dmax:
                dmax = np.nanmax(ts[p].values)
        self.gdmin = dmin
        self.gdmax = dmax

        return res


    def sort(self, key='max'):
        """
        Sort the points in each time series following the key arg
        """


class TimeSeries():

    def __init__(self, fname):
        self.fname = fname
        
        self.hdf = Dataset(fname, 'r', format='NETCDF4')
        
        ## Get dates
        self.dates = self.hdf.variables['ts_dates'][:].astype(np.float)
        self.dates[self.dates==0.] = np.nan
        self.tmin = np.nanmin(self.dates)
        self.tmax = np.nanmax(self.dates)

        # get landval sites names
        self.point_names = self.hdf.variables['point_names'][:]
   
    def use_var(self, vname):
        """
        Create a dataframe based on the selected variable.
        
        Each column is a point source.
        First column is the date.
        index is just a counter.

        """
        data = self.hdf.variables[vname][:,0,:]
        print(data.shape)
   
        #DEBUG: reduce data
        if 0:
            data = data[:,:10]
            self.point_names = self.point_names[:10]
        
        self.df = pd.DataFrame(data, columns=self.point_names)
        self.df.insert(0, 'timestamp', self.dates)
        

        # TODO later: sort data
        if 0:
            ## Sort time series using the maximum of each series
            new_order = np.argsort(np.nanmax(data, axis=1))[::-1]
            self.data = data[new_order]
            self.point_names = point_names[new_order]
    


if __name__ == "__main__":

    filelist = [i for i in sys.argv if i.endswith('.nc')] 

    tsg = TimeSeriesGroup(filelist)

    vname = sys.argv[-1]
    
    s = Sparkline(tsg)
    #s.run('wrap')
    s.run(vname, 'oneline')
    
