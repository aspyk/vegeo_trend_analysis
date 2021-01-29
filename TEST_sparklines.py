"""
Created on Fri Oct  4 09:47:45 2019

@author: moparthys

code to prepare tendencies based on Man-kendall test

"""


import numpy as np
from netCDF4 import Dataset
import sys, glob, os
import pathlib


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

from io import BytesIO

def sparkline(data, text='', figsize=(4, 0.25), **kwags):
    """
    Returns a HTML image tag containing a base64 encoded sparkline style plot
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
   
    m = 'full'
    m = 'wrap' 

    if m=='full':
        ax.plot(data)
        
        ## Add a red dot at the right hand end of the line
        #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')

        ax.fill_between(range(len(data)), data, len(data)*[np.nanmin(data)], alpha=0.1)

        ## Add horizontal line and label for max
        ax.axhline(np.nanmax(data), c='k', alpha=0.1)
        plt.text(len(data) - 1, 1.2*np.nanmax(data), '{:.2f}'.format(np.nanmax(data)), horizontalalignment='right', fontsize=8)
    
    elif m=='wrap':
        d = data.reshape((-1,36))
        for y in d:
            ax.plot(y, c='b', alpha=1./5.)
        
        ## Add a red dot at the right hand end of the line
        #plt.plot(len(data) - 1, data[len(data) - 1], 'r.')

        #ax.fill_between(range(len(data)), data, len(data)*[np.nanmin(data)], alpha=0.1)
        
        ## Add horizontal line and label for max
        ax.axhline(np.nanmax(data), c='k', alpha=0.1)
        plt.text(0, 1.2*np.nanmax(data), text[:18], horizontalalignment='left', fontsize=8)
        plt.text(36, 1.2*np.nanmax(data), '{:.2f}'.format(np.nanmax(data)), horizontalalignment='right', fontsize=8)
        ax.set_xlim(0,36)
    


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


h1 = """
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="mystyle.css">
</head>
<body>
"""
h2 = """
</body>
</html>
"""


def main():

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

    #data = data[:,50:150].T
    #data = data[:,:100].T
    data = data.T
    data = data[np.argsort(np.nanmax(data, axis=1))[::-1]]

    with open("sparklines_{}.html".format(vname), "w") as f:
        f.write(h1)
        for i in range(data.shape[0]):
            if i%50==0:
                print(i)
            ## Check if not full of nan
            if np.count_nonzero(np.isnan(data[i])) < len(data[i]):
                f.write('<div><img src="data:image/png;base64,{}"/></div>'.format(sparkline(data[i], point_names[i], figsize=(2,0.5))))
        f.write(h2)

if __name__ == "__main__":
    main()
    

