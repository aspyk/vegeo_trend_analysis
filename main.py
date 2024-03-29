#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import os,sys
import generic
import yaml
# force non-interactive backend before loading matplotlib.pyplot
# https://stackoverflow.com/a/44922799/1840524
import matplotlib
matplotlib.use('Agg')

def print_versions():
    print('Python '+sys.version)
    if sys.hexversion < 0x30501f0: # = v3.5.1 final 0
        print("ERROR: Python >= 3.5 required.")
        sys.exit()
    import numpy as np
    print("numpy =", np.__version__)
    import pandas as pd
    print("pandas =", pd.__version__)
    import matplotlib 
    print("matplotlib =", matplotlib.__version__)
    import netCDF4 
    print("netCDF4 =", netCDF4.__version__)
    import h5py.version
    print("h5py = ", h5py.version.version)
    print("hdf5 = ", h5py.version.hdf5_version)

def print_section(title):
    hline = '#----------------------------------#'
    print(hline)
    print('# '+title)
    print(hline)

class ParseInput():
    def __init__(self, input_string):
        self.type = input_string.split(':')[0]
        self.param = input_string.split(':')[1].split(',')
        if len(self.param)==1:
            self.param = self.param[0]
        
class Main():

    def __init__(self):

        if 1:
            print_section('Print versions')
            print_versions()

        self.zone_names = ['Euro', 'NAfr', 'SAfr', 'SAme', 'Fra']
        msg_vars = ['albedo', 'lai', 'evapo', 'dssf', 'lst']
        c3s_vars = ['al_bbdh','al_bbbh','al_spdh','al_spbh','lai','fapar']
        c3s_vars = ['c3s_'+i for i in c3s_vars]
        c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR','_VGT','_PROBAV','_SENTINEL3']]
        self.product_names = msg_vars + c3s_vars 



    def _parse_args(self):
        parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-t0','--start_date',
                            help='start date (ISO format %%Y-%%m-%%d) for reading of time series.',
                            #type=lambda s: datetime.date.fromisoformat(s), # on new versions of datetime
                            type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), # legacy version
                            required=True)    
        parser.add_argument('-t1','--end_date',
                            help='end date (ISO format %%Y-%%m-%%d) for reading of time series.',
                            #type=lambda s: datetime.date.fromisoformat(s), # on new versions of datetime
                            type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), # legacy version
                            required=True)    
        parser.add_argument('-p','--product_tag',
                            help='product tag or product dataset',
                            choices=self.product_names,
                            nargs='+',
                            required=True)    
        parser.add_argument('-i','--input',
                            help='input type and parameter(s), use format <type>:<param1>,<param2>... \n \
                                  box:<xmin,xmax,ymin,ymax> \n \
                                  alias:<{0}> \n \
                                  latloncsv:<path_to_csv_file>'.format('|'.join(self.zone_names)),
                            type=ParseInput,
                            required=True)    
        parser.add_argument('-a','--action',
                            help='Action to run between extract, group, trend, plot',
                            choices=['extract', 'merge', 'trend', 'join', 'plot', 'snht'],
                            nargs='+',
                            required=True)
        parser.add_argument('-n_master','--master_chunk',
                            help='size of master chunks',
                            default=500,
                            type=int)
        parser.add_argument('-f','--force_new_cache', help='Delete cache by overwritting it.', action='store_true')
        parser.add_argument('-np','--nproc', help='number of process to run the trend analysis', default=1, type=int)
        parser.add_argument('-d','--debug', help='Some debug option', default=0, type=int)
        parser.add_argument('-c','--config', help='YAML config file', required=True)
        
        if self.batch_args is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(self.batch_args)

    def preprocess(self, batch_args=None):
        """
        Process the command line options
        """
         
        print_section('Pre-processing')

        # Add some text coloring
        CR  = '\33[31m' # red
        CG  = '\33[32m' # green
        NC  = '\33[0m' # end of coloring
        # Example: print('this is in ' + CR + 'red' + NC) or print(f'this is in {CR}red{NC}')
    
        def red(s):
            C  = '\33[31m' # red
            NC = '\33[0m' # end of coloring
            return C+s+NC
    
        def green(s):
            C  = '\33[32m' # green
            NC = '\33[0m' # end of coloring
            return C+s+NC
    
        def yellow(s):
            C  = '\33[33m' # green
            NC = '\33[0m' # end of coloring
            return C+s+NC
    
    
        ## pre-process arguments
        self.batch_args = batch_args
        self._parse_args()
    
        
        dic_zone = {}
        # MSG disk: x1,x2,y1,y2, origin at the top left
        dic_zone['Euro'] = [1550, 3250,   50,  700]
        dic_zone['NAfr'] = [1240, 3450,  700, 1850]
        dic_zone['SAfr'] = [2140, 3350, 1850, 3040]
        dic_zone['SAme'] = [  40,  740, 1460, 2970]
        dic_zone['Fra']  = [1740, 2060,  310,  510]
        
        ## Read config yaml file
        with open(self.args.config, 'r') as stream:
            try:
                self.yfile = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit()
        print('*** Config file read OK.')
    
        ## Parse input format
        if self.args.input.type in ['box','alias']:
            ## Generate chunks with splitter object
            if self.args.input.type=='box':
                zone_idx = self.args.input.param
            elif self.args.input.type=='alias':
                zone_idx = dic_zone[self.args.input.param]
            self.chunks = generic.Splitter(*zone_idx)
            self.chunks.subdivide(self.args.master_chunk)
        elif self.args.input.type=='latloncsv':
            ## Loop on product
            self.products = []
            for p in self.args.product_tag:
                ## Get sensor type
                sensor = p.split('_')[-1]
                
                ## Set origin of the coordinate file
                if self.args.input.param=='config':
                    input_coor = self.yfile['ref_site_coor'][sensor]
                else:
                    input_coor = self.args.input.param=='config'
                
                ## Can use a subset of points for debugging if --debug = 1
                if self.args.debug==1:
                    slist = ['BELMANIP_00108', 'DIRECT_00039 - WatsonLake1', 'BELMANIP_00099', 'CA-NS4', 'NSA_YJP_BOREAS', 'CA-OJP', 'SS_OJP_BOREAS', 'CA-SJ1', 'CA-SJ3', 'PADDOCKWOOD', 'BELMANIP_00095', 'BELMANIP_00094', 'BELMANIP_00106', 'BRATTS_LAKE', 'REGINA', 'BELMANIP_00086']
                    #slist = ['FRENCHMAN_FLAT', 'BELMANIP_00332', 'Egypt#1', 'EL_FARAFRA', 'BELMANIP_00416', 'DOM1']
                    self.chunks = generic.CoordinatesConverter(input_coor, sensor=sensor, sub=slist)
                else: 
                    self.chunks = generic.CoordinatesConverter(input_coor, sensor=sensor)
                
                ## Finally Product objects
                prod = generic.Product(p, self.args.start_date, self.args.end_date, self.chunks)
                if prod.start_date is not None:
                    self.products.append(prod)
                    self.products[-1].infos()


        ## Update list of product name with only available one
        self.args.product_tag = [p.name for p in self.products]

        ## print info
        print("Run {4} of {0} in {1} from {2} to {3}".format(green(','.join(self.args.product_tag)),
                                                             green(','.join(self.chunks.get_limits('global', 'str'))),
                                                             green(self.args.start_date.isoformat()),
                                                             green(self.args.end_date.isoformat()), 
                                                             green(','.join(self.args.action))) )
    
    def process(self):
        """
        Run the selected pipeline actions
        """

        #------------------------------------------------#
        # EXTRACT
        #------------------------------------------------#
        if 'extract' in self.args.action:
            print_section('Extract time series')
    
            import time_series_reader
            
            cache_files = []
            for prod in self.products:
                print('  Process {0}'.format(prod.name))
                
                list_args = [prod, prod.chunks, self.yfile, self.args.force_new_cache]
                extractor = time_series_reader.TimeSeriesExtractor(*list_args)
                cache = extractor.run()
                #cache = extractor.run_snow()
                cache_files += cache
                print("cache files:")
                for c in cache_files:
                    print(c.as_posix())
    
        #------------------------------------------------#
        # MERGE
        #------------------------------------------------#
        if 'merge' in self.args.action:
            print_section('Merge time series')
    
            import time_series_merger
    
            merger = time_series_merger.TimeSeriesMerger(cache_files)
            merged_prod = merger.run()
            
            self.products = [merged_prod]

        #------------------------------------------------#
        # TREND
        #------------------------------------------------#
        if 'trend' in self.args.action:
            print_section('Compute trends')
    
            import compute_trends
    
            for prod in self.products:
                print('  Process {0}'.format(prod.name))
                
                list_args = [prod, self.chunks, self.args.nproc, self.args.force_new_cache, self.yfile]
                compute_trends.main(*list_args)
    
        #------------------------------------------------#
        # JOIN
        #------------------------------------------------#
        if 'join' in self.args.action:
            print_section('Join trends')
    
            import trend_file_merger
    
            for prod in self.args.product_tag:
                print('  Process {0}'.format(prod))
                
                list_arg = [prod, self.chunks, self.yfile, self.dic_hash[prod]]
                trend_file_merger.merge_trends(*list_arg)
    
        #------------------------------------------------#
        # PLOT
        #------------------------------------------------#
        if 'plot' in self.args.action:
            print_section('Generate plot')
    
            import trend_file_merger
    
            for prod in self.products:
                print('  Process {0}'.format(prod.name))
                
                title = '{}:{} to {}'.format(prod.hash, self.args.start_date.isoformat(), self.args.end_date.isoformat())
    
                if self.chunks.input=='box':
                    for var in ['sn','zval','pval','len']:
                        list_arg = [prod, self.chunks, title, var, self.yfile]
                        trend_file_merger.plot_trends(*list_arg)
                elif self.chunks.input=='points':
                    list_arg = [prod, self.chunks, title, self.yfile]
                    trend_file_merger.plot_trends_scatter(*list_arg)
    
        #------------------------------------------------#
        # SNHT
        #------------------------------------------------#
        if 'snht' in self.args.action:
            print_section('Compute MK test on break analysis')
    
            import TOOL_snht
    
            for prod in self.products:
                print('  Process {0}'.format(prod.name))
                    
                list_arg = [prod, self.yfile]
                TOOL_snht.QMmodule(*list_arg)

if __name__ == "__main__":
    
    if 1:
        print_section('Print versions')
        print_versions()
    
    main = Main()
    print_section('Pre-processing')
    main.preprocess()
    main.process()

