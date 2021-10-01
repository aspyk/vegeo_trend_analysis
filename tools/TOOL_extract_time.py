from datetime import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

class Processing():
    def __init__(self):
        self.parse_output()
        self.plot_hist()

    def parse_output(self):
        fname = sys.argv[-1]
        
        with open(fname, 'r') as fi:
            ## get line with time profiling
            lines = [l.split()[-2:] for l in fi.readlines() if 'load time for' in l]
        
        ## format extracted lines
        lines = [ [l[0][:-1], float(l[1])] for l in lines]
        
        ## convert to dataframe for processing
        df = pd.DataFrame(lines, columns=['var','time'])
        
        ## Get variable list
        v_unique = df['var'].unique().tolist()
        
        ## Pivot the dataframe to have columns of time by variables
        df2 = pd.DataFrame()
        for v in v_unique:
            df2[v] = df[df['var']==v]['time'].values
        
        print('### INFOS')
        df2.info()
        print('### Mean time [s]')
        print(df2.mean(axis=0).sort_values())
        self.df = df2
        self.vars = df2.mean(axis=0).sort_values().index.values

    def plot_hist(self):
        """
        Compute hist with numpy and plot with matplotlib"""

        ## Initialize an array to store histogram stats
        bin_size = 0.1
        tmp = 999
        for v in self.vars:
            mean = self.df[v].values.mean()
            if mean<tmp:
                tmp = mean
        print(tmp)
        bin_size = tmp/20

        fig, axs = plt.subplots(nrows=len(self.vars), ncols=1, sharex=True)

        for idv,v in enumerate(self.vars):
            min_edge = bin_size*np.floor(self.df[v].values.min()/bin_size)
            max_edge = bin_size*np.ceil(self.df[v].values.max()/bin_size)
            N = (max_edge-min_edge)/bin_size
            bin_list = np.linspace(min_edge, max_edge, N+1)
            
            
            h = np.histogram(self.df[v].values, bins=bin_list)[0]
            h = h/h.max()
            axs[idv].bar(0.5*(bin_list[1:]+bin_list[:-1]), h, 1.05*bin_size)
            
            #N, bins, patches = axs[idv].hist(self.df[v].values, bins=bin_list, density=True)
           
            axs[idv].text(0.9, 0.5, v, horizontalalignment='left',
                            verticalalignment='center', transform=axs[idv].transAxes, fontsize=8,
                            bbox=dict(boxstyle="square",
                                      ec=(0., 0., 0.),
                                      fc=(1., 1., 1.),)   )
            axs[idv].grid()
        
        plt.show()



if __name__=='__main__':
    proc = Processing()
