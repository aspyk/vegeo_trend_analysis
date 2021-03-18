#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os,sys
from contextlib import redirect_stdout,redirect_stderr
import argparse
import logging

from main import Main


logging.basicConfig(filename='traceback.log')

def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r','--redirect_output', help='Redirect stdout and stderr to file.', action='store_true')
param = parser.parse_args()

def run_pipeline(args, output_to='screen'):
    m = Main()
    if output_to=='file':
        sys.excepthook = exception_hook # Write exception to log file
        with open('out.txt', 'w', buffering=1) as fout:
            with redirect_stdout(fout):
                with open('err.txt', 'w', buffering=1) as ferr:
                    with redirect_stderr(ferr):
                        print("python main.py "+args)
                        m.preprocess(args.split())
                        m.process()
    elif output_to=='screen':
        print("python main.py "+args)
        m.preprocess(args.split())
        m.process()


#c3s_vars = ['al_bbdh','al_bbbh','al_spdh','al_spbh','lai','fapar']
c3s_vars = ['al_bbdh']
c3s_vars = ['c3s_'+i for i in c3s_vars]
#c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR','_VGT','_PROBAV','_SENTINEL3']]
#c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR','_VGT','_PROBAV']]
#c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR']]
c3s_vars = [i+'_AVHRR '+i+'_VGT '+i+'_PROBAV' for i in c3s_vars]

for v in c3s_vars:
    #args = "-t0 1981-09-01 -t1 1981-09-30 -i latloncsv:config -p {} -a extract -d --debug 1 --config config_vito.yml".format(v).split()
    #args = "-t0 1981-09-01 -t1 1984-12-31 -i latloncsv:config -p {} -a extract merge trend plot --debug 1 --config config_vito.yml".format(v).split()
    #args = "-t0 1981-09-01 -t1 2007-12-31 -i latloncsv:config -p {} -a extract merge trend plot --config config_vito.yml".format(v).split()
    #args = "-t0 1981-09-01 -t1 1981-12-31 -i latloncsv:config -p {} -a extract merge trend --config config_vito.yml".format(v).split()
    #args = "-t0 1981-01-01 -t1 1981-12-31 -i latloncsv:config -p {} -a extract merge trend plot -d --debug 1 --config config_vito.yml".format(v).split()
    #args = "-t0 1998-03-01 -t1 1998-05-31 -i latloncsv:config -p {} -a extract merge trend plot -d --debug 1 --config config_vito.yml".format(v).split()

    args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot --config config_vito.yml"
    #args = "-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_lai_AVHRR c3s_lai_VGT c3s_lai_PROBAV -a extract merge trend plot --config config_vito.yml"


    if param.redirect_output:
        dest = 'file'
    else:
        dest = 'screen'

    run_pipeline(args, output_to=dest)
