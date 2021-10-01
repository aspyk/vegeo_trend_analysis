import sys
print('Python '+sys.version)
if sys.hexversion < 0x30501f0: # = v3.5.1 final 0
    print("ERROR: Python >= 3.5 required.")
    sys.exit()
import numpy as np
print("numpy =", np.__version__)
import pandas as pd
print("pandas =", pd.__version__)
import matplotlib as mpl
print("matplotlib =", mpl.__version__)
import matplotlib.pyplot as plt
import netCDF4 
print("netCDF4 =", netCDF4.__version__)
import h5py
print("h5py = ", h5py.version.version)
print("hdf5 = ", h5py.version.hdf5_version)

import tools
from datetime import *


if len(sys.argv[1:])>0:
    kwargs = tools.parse_args()
else:
    kwargs ={}

bla = np.arange(12).reshape((3,4))

if 'h5file' in kwargs.keys():
    h = h5py.File(kwargs['h5file'], 'r')
    if 'h5var' in kwargs.keys():
        hv = h[kwargs['h5var']]


