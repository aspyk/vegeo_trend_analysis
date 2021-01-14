#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import os,sys
import pathlib
import generic
import yaml

class ParseInput():
    def __init__(self, input_string):
        self.type = input_string.split(':')[0]
        self.param = input_string.split(':')[1].split(',')
        if len(self.param)==1:
            self.param = self.param[0]
        

def parse_args():
    
    global zone_names

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
                        choices=['albedo', 'lai', 'evapo', 'dssf', 'lst'],
                        nargs='+',
                        required=True)    
    parser.add_argument('-i','--input',
                        help='input type and parameter(s), use format <type>:<param1>,<param2>... \n \
                              box:<xmin,xmax,ymin,ymax> \n \
                              alias:<{0}> \n \
                              latloncsv:<path_to_csv_file>'.format('|'.join(zone_names)),
                        type=ParseInput,
                        required=True)    
    parser.add_argument('-a','--action',
                        help='Action to run between extract, group, trend, plot',
                        choices=['extract', 'append', 'trend', 'merge', 'plot'],
                        nargs='+',
                        required=True)
    parser.add_argument('-n_master','--master_chunk',
                        help='size of master chunks',
                        default=500,
                        type=int)
    parser.add_argument('-d','--delete_cache', help='Delete cache by overwritting it.', action='store_true')
    parser.add_argument('-np','--nproc', help='number of process to run the trend analysis', default=1, type=int)
    
    args = parser.parse_args()
    return args


def main():

    ######################
    ### PRE-PROCESSING ###
    ######################

    global zone_names

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
    args = parse_args()

    
    dic_zone = {}
    # MSG disk: x1,x2,y1,y2, origin at the top left
    dic_zone['Euro'] = [1550, 3250,   50,  700]
    dic_zone['NAfr'] = [1240, 3450,  700, 1850]
    dic_zone['SAfr'] = [2140, 3350, 1850, 3040]
    dic_zone['SAme'] = [  40,  740, 1460, 2970]
    dic_zone['Fra']  = [1740, 2060,  310,  510]
    
    ## Parse input format
    if args.input.type in ['box','alias']:
        ## Generate chunks with splitter object
        if args.input.type=='box':
            zone_idx = args.input.param
        elif args.input.type=='alias':
            zone_idx = dic_zone[args.input.param]
        chunks = generic.Splitter(*zone_idx)
        chunks.subdivide(args.master_chunk)
    elif args.input.type=='latloncsv':
        chunks = generic.CoordinatesConverter(args.input.param, resol='1km')
   
   
    # print info
    print("Run {4} of {0} in {1} from {2} to {3}".format(green(','.join(args.product_tag)),
                                                         green(','.join(chunks.get_limits('global', 'str'))),
                                                         green(args.start_date.isoformat()),
                                                         green(args.end_date.isoformat()), 
                                                         green(','.join(args.action))) )

    dic_hash = {}
    for p in args.product_tag:
        dic_hash[p] = generic.get_case_hash(p, args.start_date, args.end_date, chunks)
        print(p, dic_hash[p] )

    ## Read config yaml file
    with open("config.yml", 'r') as stream:
        try:
            yfile = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()


    ##################
    ### PROCESSING ###
    ##################

    ## Run the selected pipeline actions
   
    #------------------------------------------------#
    # EXTRACT
    #------------------------------------------------#
    if 'extract' in args.action:
        print('>>> '+yellow('Extract raw data')+'...')

        import time_series_reader

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            
            list_args = [prod, args.start_date, args.end_date, chunks, yfile, dic_hash[prod]]
            extractor = time_series_reader.TimeSeriesExtractor(*list_args)
            extractor.run()

    #------------------------------------------------#
    # TREND
    #------------------------------------------------#
    if 'trend' in args.action:
        print('>>> '+yellow('Compute trends')+'...')

        import estimate_trends_from_time_series

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            
            list_args = [prod, chunks, args.nproc, args.delete_cache, yfile, dic_hash[prod]]
            estimate_trends_from_time_series.compute_trends(*list_args)

    #------------------------------------------------#
    # MERGE
    #------------------------------------------------#
    if 'merge' in args.action:
        print('>>> '+yellow('Merge trends')+'...')

        import trend_file_merger

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            
            list_arg = [prod, chunks, yfile, dic_hash[prod]]
            trend_file_merger.merge_trends(*list_arg)

    #------------------------------------------------#
    # PLOT
    #------------------------------------------------#
    if 'plot' in args.action:
        print('>>> '+yellow('Generate plot')+'...')

        import trend_file_merger

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            
            phash = dic_hash[prod]
            title = '{}:{} to {}'.format(phash, args.start_date.isoformat(), args.end_date.isoformat())

            for var in ['sn','zval','pval','len']:
                list_arg = [prod, chunks, title, var, 365, yfile, phash]
                trend_file_merger.plot_trends(*list_arg)


if __name__ == "__main__":
    global zone_names
    zone_names = ['Euro', 'NAfr', 'SAfr', 'SAme', 'Fra']
    main()

