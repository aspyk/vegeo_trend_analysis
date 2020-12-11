import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib
import subprocess
from string import Template
import time
from timeit import default_timer as timer
import glob
import os,sys
import datetime
import fnmatch



class Timeline():
    """
    TODO: merge continuous interval between end of the month and the start of the following
    """

    def __init__(self, path_template, freq):
        self.path_template = pathlib.Path(path_template)
        print(self.path_template.parts)
       
        ## Split fixed and changing part
        self.fixed_path = []
        self.changing_path = []
        for p in self.path_template.parts:
            if '$' not in p:
                self.fixed_path.append(p)
            else:
                self.changing_path.append(p)
        self.fixed_path = pathlib.Path(*self.fixed_path)
        self.changing_path = pathlib.Path(*self.changing_path)
        print('fixed:', self.fixed_path)
        print('changed:', self.changing_path)
        self.changing_path = Template(str(self.changing_path))
    

        self.freq = freq

    def _find_date(self, dic_date):
        """
        *** DEPRECATED ***
        Keep just as an example to pipe subprocesses
        """
        # Get the result of:
        # find <fixed_path> -name <changing_path> | head -5
        CMD = ['find', str(self.fixed_path), '-name', self.changing_path.substitute(**dic_date)]
        find = subprocess.Popen(CMD, stdout=subprocess.PIPE)
        res = subprocess.run(['head', '-5'], stdin=find.stdout, stdout=subprocess.PIPE)
        out = res.stdout.decode('utf-8').split('\n')[:-1] # remove last empty item
        return out
 

    def grouper(self, input_list, n=2):
        """
        Generate tuple of n consecutive items from input_list
        ex for n=2: [a,b,c,d] -> [[a,b],[b,c],[c,d]]
        """
        for i in range(len(input_list) - (n - 1)):
            yield input_list[i:i+n]


    def get_continuous_intervals(self):
        """
        Scan the provided directory to create a list of continuous intervals based on selected frequency:
        [ [t0,t1] , [t2,t3] , ... ]
        ti being datetime objects.
        """
       
        ## date template
        dic_date = {'Y':'????', 'm':'??', 'd':'??', 'H':'??', 'M':'??'}
        
        init_year = 2000
        self.valid_year = {}

        ## INFO: timing for 337055 files to list:
        # from timeit import default_timer as timer
        # t0 = timer()
        # out = list(self.fixed_path.glob(self.changing_path.substitute(**dic_date)))
        # print(len(out), timer()-t0)
        # 
        # t0 = timer()
        # out = fnmatch.filter(os.listdir(self.fixed_path), self.changing_path.substitute(**dic_date))
        # print(len(out), timer()-t0)
        # 
        # Results:
        # 337055 1.685467129573226
        # 337055 0.328439112752676
        ## > 5x faster with fnmatch

        

        ## First select available (valid) years
        ## Use os.listdir() and fnmatch.filter() to perform quick selection
        flist = os.listdir(self.fixed_path)
        for y in range(init_year, 2021):
            dic_date['Y'] = y
            #print('sub:', self.changing_path.substitute(**dic_date))
            out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))
            if len(out) > 0:
                self.valid_year[y] = []
        print("valid year:", self.valid_year.keys())


        ## Then iterate over months of valid years
        t0 = timer()
        for y in self.valid_year.keys():
            dic_date['Y'] = y
            for m in range(1,13):
                dic_date['m'] = str(m).zfill(2)
                #print('sub:', self.changing_path.substitute(**dic_date))
                out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))
                # if there is data in the month
                if len(out)>0:
                    
                    t00 = timer()
                    
                    ## Get available file by month 
                    out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))
                    # convert string template to date formater
                    ref = self.changing_path.template.replace('$','%')
                    ref = ref.replace('{','')
                    ref = ref.replace('}','')
                    avail_date = np.array(sorted([datetime.datetime.strptime(f, ref) for f in out]))
                   
                    ## Get all theoretical dates, actually not useful here
                    ## INFO: Note the use of MonthEnd to offset of one (or more)  month precisely
                    #start = datetime.datetime(y, m, 1)
                    #all_date = pd.date_range(start, start+pd.tseries.offsets.MonthEnd(1), freq=self.freq)

                    print(y, m, len(avail_date))

                    ## Recorf all valid intervalls as [start_dat, end_date] of continuous range according to self.freq
                    inter = [avail_date[0]]
                    for d1,d2 in self.grouper(avail_date):
                        # check if there is more than one self.freq between two following date
                        if d2-d1!=pd.to_timedelta(self.freq):
                            inter.append(d1)
                            self.valid_year[y].append(inter)
                            inter = [d2]
                    inter.append(avail_date[-1])
                    self.valid_year[y].append(inter)


                    #for i in valid_intervals:
                    #    print(i)


                    print(timer()-t00)

        print(timer()-t0)
                    
    def plot_intervals(self):                
        """Generate and save plot of valid interval"""
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # You can then convert these datetime.datetime objects to the correct
        # format for matplotlib to work with.

        #xlims = [mdates.date2num(datetime.datetime(2004,1,1,0,0,0)), mdates.date2num(datetime.datetime(2004,12,31,23,59,59))]
        # plot 
        #ax.imshow(res_h.T, aspect='auto', origin='lower', extent=(xlims[0], xlims[1], dmin, dmax))
        for y,inter in self.valid_year.items():
            xnum = [[mdates.date2num(j)-mdates.datestr2num('{}-01-01 00:00'.format(y)) for j in i] for i in inter]
            for x in xnum:
                ax.plot(x,[y,y], lw=10)
        
        ax.grid()

        ax.set_xlim(0,367)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()

        ax.set_xlabel('date')
        ax.set_ylabel('product')
        plt.savefig('res_timeline.png')
        
            


if __name__=="__main__":

    param = {}

    #param['path_template'] = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
    param['path_template'] = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
    param['freq'] = '15T'
    
    param['path_template'] = '/cnrm/vegeo/SAT/DATA/MSG_LAI_DAILY_CDR/HDF5_LSASAF_MSG_LAI_MSG-Disk_$Y$m${d}0000'
    param['freq'] = '1D'
    

    timeline = Timeline(**param)
    timeline.get_continuous_intervals()
    timeline.plot_intervals()

