from datetime import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mstats
import mankendall_fortran_repeat_exp2 as m
import seasonalkendall as sk

freq = 365

x = np.arange(freq*10)/(1.*freq)

y = np.sin(2*np.pi*x)

slope = 0.2 # [1/year]

# sinus + slope
y += slope*x
# random + slope only
#y = 0.1*np.random.rand(len(x)) + slope*x


#plt.plot(x,y)
#plt.show()


p,z,Sn,nx = m.mk_trend(len(y), np.arange(len(y)), y)
print('p, z, Sn, nx:')
print(p, z, Sn, nx)


slope2, intercept, lo_slope, up_slope = mstats.theilslopes(y)
print('slope2, intercept, lo_slope, up_slop:')
print(slope2, intercept, lo_slope, up_slope)

res_smk = sk.seakeni(y, 365)
print(res_smk)

print('Summary:')
print("mk fortran : {}, err[%] = {:.2f}".format(Sn*freq, 100*(slope-Sn*freq)/slope))
print("mk scipy   : {}, err[%] = {:.2f}".format(slope2*freq, 100*(slope-slope2*freq)/slope))
print("smk fortran: {}, err[%] = {:.2e}".format(res_smk[1], 100*(slope-res_smk[1])/slope))
