import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
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
    def __init__(self):
        self.path_template = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
        #self.path_template = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
        self.path_template = pathlib.Path(self.path_template)
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
    

        self.freq = '15T'

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


    def get_initial_date(self):
        
        dic_date = {'Y':'????', 'm':'??', 'd':'??', 'H':'??', 'M':'??'}
        
        init_year = 2000
        valid_year = {}

        ## timing for 337055 files to list:
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

        
        flist = os.listdir(self.fixed_path)

        ## Iterate over years
        for y in range(init_year, 2021):
            dic_date['Y'] = y
            #print('sub:', self.changing_path.substitute(**dic_date))
            out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))

            if len(out) > 0:
                valid_year[y] = []

        print("valid year:", valid_year.keys())


        ## Iterate over months of valid years
        t0 = timer()
        for y in valid_year.keys():
            dic_date['Y'] = y
            for m in range(1,13):
                dic_date['m'] = str(m).zfill(2)
                #print('sub:', self.changing_path.substitute(**dic_date))
                out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))
                if len(out)>0:
                    
                    t00 = timer()
                    
                    ## Get available file by month 
                    out = fnmatch.filter(flist, self.changing_path.substitute(**dic_date))
                    ref = self.changing_path.template.replace('$','%')
                    avail_date = np.array(sorted([datetime.datetime.strptime(f, ref) for f in out]))
                   
                    ## Get all theoretical dates, actually not useful here
                    ## Note the use of MonthEnd to offset of one (or more)  month precisely
                    #start = datetime.datetime(y, m, 1)
                    #all_date = pd.date_range(start, start+pd.tseries.offsets.MonthEnd(1), freq=self.freq)

                    print(y, m, len(avail_date))

                    valid_intervals = []
                    inter = [avail_date[0]]
                    for d1,d2 in self.grouper(avail_date):
                        if d2-d1!=pd.to_timedelta(self.freq):
                            inter.append(d1)
                            valid_intervals.append(inter)
                            inter = [d2]
                    inter.append(avail_date[-1])
                    valid_intervals.append(inter)


                    for i in valid_intervals:
                        print(i)


                    print(timer()-t00)


        print(timer()-t0)
                    
                

            

    def read_time_series(self):
        pass

if __name__=="__main__":

    timeline = Timeline()
    timeline.get_initial_date()
    #timeline.read_time_series()

