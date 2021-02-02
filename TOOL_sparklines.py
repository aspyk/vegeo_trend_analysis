import numpy as np
from netCDF4 import Dataset
import sys, glob, os
import pathlib
from datetime import datetime

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


    def __init__(self):
        self.fname_param = []
        pass

    def make_graph(self, data, mode='wrap', text='', figsize=(4, 0.25), **kwags):
        """
        Returns a HTML image tag containing a base64 encoded sparkline style plot

        Parameters
        ----------
        mode: string, 'oneline' or 'wrap'
        """
    
        dmin = np.nanmin(data)
        dmax = np.nanmax(data)

        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
       
        if mode=='oneline':

            ## Add sensor time range
            for k,v in self.dic_sen.items():
                #ax.plot(v, mx)
                ax.fill_between(v, 2*[dmax], 2*[dmin], alpha=0.1)
                # Plot name
                plt.text(0.5*(v[1]+v[0]), 1.2*dmax, k, horizontalalignment='center', fontsize=8)
            
            ## Plot sparkline
            ax.plot(self.dates, data, c=u'#1f77b4')
            ax.fill_between(range(len(data)), data, len(data)*[dmin], alpha=0.1)
 
            ## Add a red dot at the right hand end of the line
            #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    
            ## Add horizontal line and label for max
            ax.axhline(dmax, c='k', alpha=0.1)
            # Plot name
            plt.text(self.tmin, 1.2*dmax, text[:18], horizontalalignment='left', fontsize=8)
            # Plot max
            plt.text(self.tmax, 1.2*dmax, '{:.2f}'.format(dmax), horizontalalignment='right', fontsize=8)

            # Add vertical lines for years
            #for i in range(int(len(data)/36)):
            #    ax.axvline(i+1, c='k', alpha=0.1)

            ax.set_xlim(self.tmin, self.tmax)

            ## TODO
            # - plot vline for years
            # - plot limit between sensor
            # - plot several files using dates


        
        elif mode=='wrap':
            d = data.reshape((-1,36))
            for y in d:
                ax.plot(y, c='b', alpha=1./5.)
            
            ## Add a red dot at the right hand end of the line
            #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    
            #ax.fill_between(range(len(data)), data, len(data)*[np.nanmin(data)], alpha=0.1)
            
            ## Add horizontal line and label for max
            ax.axhline(dmax, c='k', alpha=0.1)
            # Plot name
            plt.text(0, 1.2*dmax, text[:18], horizontalalignment='left', fontsize=8)
            # Plot max
            plt.text(36, 1.2*dmax, '{:.2f}'.format(dmax), horizontalalignment='right', fontsize=8)
            ax.set_xlim(0,36)
        
        ## Remove all ticks and frame to make sparkline style 
        if 1: 
            for k,v in ax.spines.items():
                v.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
    
    
        #plt.title('Mn/Mx: {:.2f} / {:.2f}'.format(np.nanmin(data), np.nanmax(data)), fontsize=8)
        
    
        img = BytesIO()
        plt.savefig(img, transparent=True, bbox_inches='tight')
        img.seek(0)
        plt.close()
    
        return base64.b64encode(img.read()).decode("UTF-8")    
    
    
   
    
    def run(self, mode, figsize):
    
        hdf_ts = Dataset(sys.argv[1], 'r', format='NETCDF4')
        fname = Dataset('output_extract/c3s_alspdh/b32a16_timeseries_0_725_0_1.nc', 'r', format='NETCDF4')
        
        # get data
        if len(sys.argv) > 2:
            vname = sys.argv[2]
        else:
            vname = 'time_series_chunk'
        data = hdf_ts.variables[vname][:,0,:]
        # get landval sites names
        point_names = fname.variables['point_names'][:]
    
    
        print(data.shape)
    
        #data = data[:,:10].T
        data = data.T
        ## Sort time series using the maximum of each series
        data = data[np.argsort(np.nanmax(data, axis=1))[::-1]]
    
        ## Get dates
        self.dates = hdf_ts.variables['ts_dates'][:].astype(np.float)
        self.dates[self.dates==0.] = np.nan
        self.tmin = np.nanmin(self.dates)
        self.tmax = np.nanmax(self.dates)

        ## Param to output in filename
        self.fname_param.append(vname)
        self.fname_param.append(mode)

        with open("sparklines_{}.html".format('_'.join(self.fname_param)), "w") as f:
            f.write(self.h1)
            for i in range(data.shape[0]):
                if i%50==0:
                    print(i)
                ## Check if not full of nan
                if np.count_nonzero(np.isnan(data[i])) < len(data[i]):
                     img = self.make_graph(data[i], mode, point_names[i], figsize=figsize)
                     f.write('<div><img src="data:image/png;base64,{}"/></div>'.format(img))
            f.write(self.h2)

if __name__ == "__main__":
    s = Sparkline()
    #s.run('wrap', (2,0.5))
    s.run('oneline', (18,0.5))
    
