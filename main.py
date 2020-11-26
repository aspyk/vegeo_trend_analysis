#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import datetime
import os,sys
import pathlib
import hashlib
import generic


def parse_args():
    
    global zone_names

    parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t0','--start_date',
                        help='start date (ISO format %%Y-%%m-%%d) for reading of time series.',
                        type=lambda s: datetime.date.fromisoformat(s),
                        required=True)    
    parser.add_argument('-t1','--end_date',
                        help='end date (ISO format %%Y-%%m-%%d) for reading of time series.',
                        type=lambda s: datetime.date.fromisoformat(s),
                        required=True)    
    parser.add_argument('-o','--output', help='output path')    
    parser.add_argument('-p','--product_tag',
                        help='product tag or product dataset',
                        choices=['albedo', 'lai', 'evapo', 'dssf', 'fapar', 'lst'],
                        nargs='+',
                        required=True)    
    groupz = parser.add_mutually_exclusive_group(required=True)
    groupz.add_argument('-zc','--zone_coor',
                        help='Zone to be analysed. Given as <xmin xmax ymin ymax> as the boundary box coordinates.',
                        nargs='+',
                        type=int) 
    groupz.add_argument('-zn','--zone_name',
                        help='Zone to be analysed. Given as a name in {0}.'.format(', '.join(zone_names)),
                        choices=zone_names)
    parser.add_argument('-a','--action',
                        help='Action to run between extract, group, trend, plot',
                        choices=['extract', 'append', 'trend', 'merge', 'plot'],
                        nargs='+',
                        required=True)
    parser.add_argument('-n_master','--master_chunk',
                        help='size of master chunks',
                        default=500,
                        type=int)
    
    args = parser.parse_args()
    return args


def main():

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


    # pre-process arguments
    args = parse_args()


    start_year = int(args.start_date.year)
    end_year = int(args.end_date.year)
    
    start_month = int(args.start_date.month)
    end_month = int(args.end_date.month)
    
    dic_zone = {}
    # x1,x2,y1,y2, origin at the top left
    dic_zone['Euro'] = np.array([1550, 3250,   50,  700])
    dic_zone['NAfr'] = np.array([1240, 3450,  700, 1850])
    dic_zone['SAfr'] = np.array([2140, 3350, 1850, 3040])
    dic_zone['SAme'] = np.array([  40,  740, 1460, 2970])
    dic_zone['Fra']  = np.array([1740, 2060,  310,  510])
    
    if args.zone_name is not None:
        args.zone_coor = dic_zone[args.zone_name]
   
    

    # Set the boundary box coordinates
    # Note: swap to fit numpy array indexing: array[y1:y2,x1:x2]
    # Just a patch here, to be checked later
    xlim1 = args.zone_coor[2] 
    xlim2 = args.zone_coor[3]
    ylim1 = args.zone_coor[0]
    ylim2 = args.zone_coor[1]

    nmaster = args.master_chunk
   
    # Generate chunks with splitter object
    chunks = generic.Splitter(xlim1, xlim2, ylim1, ylim2, nmaster)

    # print info
    print("Run {4} of {0} in {1} from {2} to {3}".format(green(','.join(args.product_tag)),
                                                         green(chunks.global_limits(fmt='str')),
                                                         green(args.start_date.isoformat()),
                                                         green(args.end_date.isoformat()), 
                                                         green(','.join(args.action))) )

    ## test to use a hash
    str_hash = '{0},{1},{2}'.format(args.start_date.isoformat(), args.end_date.isoformat(), chunks.global_limits(fmt='str'))
    print(str_hash)
    hex_hash = hashlib.sha1(str_hash.encode("UTF-8")).hexdigest()
    print(hex_hash[:6])
    print(hex_hash[:8])


    ## Run the selected pipeline actions
   
    # EXTRACT
    #------------------------------------------------#
    if 'extract' in args.action:
        print('>>> '+yellow('Extract raw data')+'...')

        import time_series_reader

        if args.output==None:
            list_args = [args.start_date, args.end_date, None, chunks]
        else:
            list_args = [args.start_date, args.end_date, None, chunks, args.output]

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            list_args[2] = prod
            extractor = time_series_reader.TimeSeriesExtractor(*list_args)
            extractor.run()

    # TREND
    #------------------------------------------------#
    if 'trend' in args.action:
        print('>>> '+yellow('Compute trends')+'...')

        import estimate_trends_from_time_series

        if args.output==None:
            output_path = './output_tendencies/'
            input_path = './output_timeseries/'
        else:
            output_path = args.output

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            pathlib.Path(output_path+'/'+prod).mkdir(parents=True, exist_ok=True)

            str_arg = [str(i) for i in [start_year, end_year, start_month, end_month, input_path, output_path, prod, xlim1, xlim2, ylim1, ylim2, nmaster]]
            estimate_trends_from_time_series.compute_trends(*str_arg)

    # MERGE
    #------------------------------------------------#
    if 'merge' in args.action:
        print('>>> '+yellow('Merge trends')+'...')

        import trend_file_merger

        input_path = './output_tendencies/'

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            
            input_path += '/'+prod 
            input_path = os.path.normpath(input_path) + os.sep

            merged_filename = 'merged_trends.nc'

            list_arg = [input_path, merged_filename, xlim1, xlim2, ylim1, ylim2, nmaster]
            trend_file_merger.merge_trends(*list_arg)

    # PLOT
    #------------------------------------------------#
    if 'plot' in args.action:
        print('>>> '+yellow('Generate plot')+'...')

        import trend_file_merger

        if args.output==None:
            output_path = './output_plots/'
            input_path = './output_tendencies/'
        else:
            output_path = args.output

        for prod in args.product_tag:
            print('  Process {0}'.format(yellow(prod)))
            pathlib.Path(output_path+'/'+prod).mkdir(parents=True, exist_ok=True)
            
            input_path += '/'+prod 
            input_path = os.path.normpath(input_path) + os.sep

            output_path += '/'+prod 
            output_path = os.path.normpath(output_path) + os.sep
           
            merged_filename = 'merged_trends.nc'

            for var in ['sn','zval','pval','len']:
                list_arg = [input_path, output_path, merged_filename, xlim1, xlim2, ylim1, ylim2,
                            '{}:{} to {}'.format(hex_hash[:6], args.start_date.isoformat(), args.end_date.isoformat()), var, 365]
                trend_file_merger.plot_trends(*list_arg)


if __name__ == "__main__":
    global zone_names
    zone_names = ['Euro', 'NAfr', 'SAfr', 'SAme', 'Fra']
    main()

