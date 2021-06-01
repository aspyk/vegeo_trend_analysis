#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os,sys
from contextlib import redirect_stdout,redirect_stderr
import argparse
import logging
from pprint import pprint

from main import Main

tstamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S.log")

logging.basicConfig(filename='traceback'+tstamp)

def print_section(title):
    """Format section title"""
    hline = '#----------------------------------#'
    print(hline)
    print('# '+title)
    print(hline)

def exception_hook(exc_type, exc_value, exc_traceback):
    """Redirect uncaught exception to a specific log file."""
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


def redirect(param, output_to='screen'):
    """
    Redirect output either to screen or to a file.
    """
    if output_to=='file':
        print("Write output to", 'out'+tstamp)
        print("Started at {}".format(datetime.datetime.now()))
        sys.excepthook = exception_hook # Write exception to log file
        with open('out'+tstamp, 'w', buffering=1) as fout:
            with redirect_stdout(fout):
                with open('err'+tstamp, 'w', buffering=1) as ferr:
                    with redirect_stderr(ferr):
                        main(param)
        print("Finished at {}".format(datetime.datetime.now()))
    elif output_to=='screen':
        main(param)

def run_pipeline(args):
    print("python main.py "+args)
    m = Main()
    m.preprocess(args.split())
    m.process()

def get_product_list(products, sensors):
    """Make the product tags"""
    res = []
    for p in products:
        #res.append([i+r for i in c3s_vars for r in ['_AVHRR','_VGT','_PROBAV','_SENTINEL3']]
        res.append(['c3s_'+p+'_'+s for s in sensors])
    return res

def make_cmd_args(products, start='1981-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'),
                  actions=['extract', 'merge', 'trend', 'plot'], conf_file='config_vito.yml', extra_flags=''):
    """Make the command line for the pipeline"""
    dic = {}
    dic['-t0'] = start
    dic['-t1'] = end
    dic['-i'] = 'latloncsv:config'
    dic['-p'] = ' '.join(products)
    dic['-a'] = ' '.join(actions)
    dic['--config'] = 'config_vito.yml'

    print_section('Summary')
    pprint(dic)

    res = ' '.join([k+' '+v for k,v in dic.items()]) + ' ' + extra_flags

    return res

def main(param):
    ## Make product tags
    prod = get_product_list(products=param.product, sensors=param.sensor)
    
    ## Run pipeline for each product
    extra = ''
    if param.debug: extra += '--debug 1 '
    if param.force_new_cache: extra += '-f '
    
    for p in prod:
        args = make_cmd_args(p, extra_flags=extra)
    
        print(args)
    
        #args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot --config config_vito.yml"
        #args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbbh_AVHRR c3s_al_bbbh_VGT c3s_al_bbbh_PROBAV -a extract merge trend plot --config config_vito.yml"
        #args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_spbh_AVHRR c3s_al_spbh_VGT c3s_al_spbh_PROBAV -a extract merge trend plot --config config_vito.yml"
        #args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_spdh_AVHRR c3s_al_spdh_VGT c3s_al_spdh_PROBAV -a extract merge trend plot --config config_vito.yml"
    
        run_pipeline(args)


if __name__=='__main__':

    ## Parse command line arguments
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p','--product',
                        help='Product(s) to analyze',
                        choices=['al_bbdh','al_bbbh','al_spdh','al_spbh','lai','fapar'],
                        nargs='+',
                        required=True)    
    parser.add_argument('-s','--sensor',
                        help='Sensor(s) to be analyzed',
                        choices=['AVHRR','VGT','PROBAV','SENTINEL3'],
                        nargs='+',
                        required=True)    
    parser.add_argument('-r','--redirect_output', help='Redirect stdout and stderr to file.', action='store_true')
    parser.add_argument('-d','--debug', help='Debug mode: read only some reference sites (names has to be givent in main.py)', action='store_true')
    parser.add_argument('-f','--force_new_cache', help='If existing, previous cache files are not read, all the pipeline is reprocessed. Note that previous cache files may be overwritten when new cache files are produced.', action='store_true')
    param = parser.parse_args()
    
    if param.redirect_output:
        dest = 'file'
    else:
        dest = 'screen'
    
    redirect(param, output_to=dest)
