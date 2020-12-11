import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from matplotlib.collections import PolyCollection
import pathlib
import subprocess
from string import Template
import time
from timeit import default_timer as timer
import glob
import os,sys
import datetime
import fnmatch
import itertools



class Timeline():
    """
    TODO: merge continuous interval between end of the month and the start of the following
    """

    def __init__(self, prod, path_template, freq):
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
        self.file_template = self.changing_path.name
        print('fixed:', self.fixed_path)
        print('changed:', self.changing_path)
        print('file:', self.file_template)
        self.changing_path = Template(str(self.changing_path))
        self.file_template = Template(str(self.file_template))
    
        self.freq = freq
        self.prod = prod

        self.init_year = 2000
        self.last_year = 2020
        #self.init_year = 2016
        #self.last_year = 2017

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
        t0 = timer()
        #flist = os.listdir(self.fixed_path)
        flist = [ f for dp, dn, fn in os.walk(self.fixed_path) for f in fn]
        print(timer()-t0)
        #return
        for y in range(self.init_year, self.last_year+1):
            dic_date['Y'] = y
            #print(y, 'sub:', self.file_template.substitute(**dic_date))
            out = fnmatch.filter(flist, self.file_template.substitute(**dic_date))
            if len(out) > 0:
                self.valid_year[y] = []
        print("valid year:", self.valid_year.keys())

        #return

        ## Then iterate over months of valid years
        t0 = timer()
        for y in self.valid_year.keys():
            dic_date['Y'] = y
            for m in range(1,13):
                dic_date['m'] = str(m).zfill(2)
                #print('sub:', self.changing_path.substitute(**dic_date))
                out = fnmatch.filter(flist, self.file_template.substitute(**dic_date))
                out = [i for i in out if not i.endswith('.bz2')]
                # if there is data in the month
                if len(out)>0:
                    
                    t00 = timer()
                    
                    ## Get available file by month 
                    #convert string template to date formater
                    ref = self.file_template.template.replace('$','%')
                    ref = ref.replace('{','')
                    ref = ref.replace('}','')
                    # hack to add custom parsing in strptime
                    # https://stackoverflow.com/a/54451291/1840524
                    import _strptime
                    TimeRE = _strptime.TimeRE()
                    TimeRE.update({'x': '(.h5)?'}) # match 0 or 1 '.h5' pattern
                    _strptime._TimeRE_cache = TimeRE
                    ref = ref.replace('*','%x')
                    # process input list
                    # remove bz2 files
                    avail_date = np.array(sorted([datetime.datetime.strptime(f, ref) for f in out]))
                   
                    ## INFO: Get all theoretical dates, actually not useful here
                    ## Note the use of MonthEnd to offset of one (or more)  month precisely
                    #start = datetime.datetime(y, m, 1)
                    #all_date = pd.date_range(start, start+pd.tseries.offsets.MonthEnd(1), freq=self.freq)

                    print(y, m, len(out), '>', len(avail_date))

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



                    #print(timer()-t00)

        print(timer()-t0)

        self.show_tot_time()

        ## Merge continuous intervals when changing month.
        for y,inter in self.valid_year.items():
            merged = []
            merged.append(inter[0])
            for i1,i2 in self.grouper(inter):
                if i2[0]-i1[1]==pd.to_timedelta(self.freq):
                    merged[-1][-1] = i2[1] # [d1,d2],[d3,d4] -> [d1,d4]
                else:
                    merged.append(i2)
            self.valid_year[y] = merged


        self.show_tot_time()
 

    def show_tot_time(self):
        """get total time / year"""
        dic_time = {}
        for y,inter in self.valid_year.items():
            dic_time[y] = []
            for i in inter:
                dic_time[y].append((i[1]-i[0]).total_seconds())

        for y,inter in dic_time.items():
            tmp = np.array(inter)
            print('{} : inter nb {} : {} s : {:.3f} days'.format(y, tmp.size, tmp.sum(), tmp.sum()/86400))


    def plot_intervals(self):                
        verts = []
        colors = []
        pool = itertools.cycle(plt.cm.Set1(np.linspace(0, 1, 9)))

        for y,inter in self.valid_year.items():
            offset = mdates.datestr2num('{}-01-01 00:00'.format(y))
            for d in inter:
                # Increase small interval to be visible on the plot
                if d[1]-d[0] < datetime.timedelta(hours=12):
                    d[1] += datetime.timedelta(hours=12)
                v =  [(mdates.date2num(d[0])-offset, y-.4),
                      (mdates.date2num(d[0])-offset, y+.4),
                      (mdates.date2num(d[1])-offset, y+.4),
                      (mdates.date2num(d[1])-offset, y-.4),
                      (mdates.date2num(d[0])-offset, y-.4)]
                verts.append(v)
                colors.append(next(pool))
        
        bars = PolyCollection(verts, facecolors=colors)
        #bars = PolyCollection(verts)

        fig = plt.figure(figsize=(20,8))
        ax = fig.add_subplot(111)
        
        ax.add_collection(bars)

        ax.grid()
        
        # set dates limits
        plt.axis('tight')
        ax.set_xlim(0,366)
        years = np.array(list(self.valid_year.keys()))
        ax.set_ylim(years.min()-0.5, years.max()+0.5)

        ax.set_xlabel('date')
        ax.set_ylabel(self.prod)
        
        #plt.gca().set_position([0, 0, 1, 1])

        fout = 'res_timeline_{}.png'.format(self.prod)
        plt.savefig(fout)
        print("Image saved to:", fout)


def get_param(prod):
    param = {}
    param['prod'] = prod

    if prod=='lst_missing':
        param['path_template'] = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_MISSING_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
        param['freq'] = '15T'
    
    elif prod=='lst_nrt':
        param['path_template'] = "/cnrm/vegeo/SAT/DATA/MSG_LST_CDR_OR_NRT_15min/HDF5_LSASAF_MSG_LST_MSG-Disk_$Y$m$d$H$M"
        param['freq'] = '15T'
    
    elif prod=='lai':
        param['path_template'] = '/cnrm/vegeo/SAT/DATA/MSG_LAI_DAILY_CDR/HDF5_LSASAF_MSG_LAI_MSG-Disk_${Y}${m}${d}0000'
        param['freq'] = '1D'
    
    elif prod=='evapo':
        param['path_template'] = '/cnrm/vegeo/SAT/DATA/LSA_SAF_METREF_CDR_DAILY/HDF5_LSASAF_MSG_METREF_MSG-Disk_$Y$m${d}0000'
        param['freq'] = '1D'

    elif prod=='al_icare':
        # ICARE: 2005-01-01 -> 2016-09-18 (~2005-2016)
        param['path_template'] = '/cnrm/vegeo/SAT/DATA/AERUS_GEO/Albedo_v104/${Y}/SEV_AERUS-ALBEDO-D3_${Y}-${m}-${d}_V1-04.h5'
        param['freq'] = '1D'
            
    elif prod=='al_mdal':
        # MDAL: 2004-01-19 -> 2015-12-31 (~2004-2015)
        param['path_template'] = '/cnrm/vegeo/SAT/DATA/MSG/Reprocessed-on-2017/MDAL/${Y}/${m}/${d}/HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_${Y}${m}${d}0000'
        param['freq'] = '1D'
       
    elif prod=='al_mdal_nrt':
        # MDAL_NRT: 2015-11-11 -> today (~2016-today)
        # 2019 -> 2020 : HDF5_xxx0000
        # 2015 -> 2018 : HDF5_xxx0000.h5
        param['path_template'] = '/cnrm/vegeo/SAT/DATA/MSG/NRT-Operational/AL2/AL2-${Y}${m}${d}/HDF5_LSASAF_MSG_ALBEDO_MSG-Disk_${Y}${m}${d}0000*'
        param['freq'] = '1D'
 
    return param


if __name__=="__main__":

    plist = ['lst_nrt', 'lst_missing', 'lai', 'evapo', 'al_icare', 'al_mdal', 'al_mdal_nrt']

    #for p in plist:
    for p in [plist[-1]]:
        param = get_param(p)

        timeline = Timeline(**param)
        timeline.get_continuous_intervals()
        timeline.plot_intervals()

