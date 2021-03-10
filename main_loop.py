#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os,sys

from main import Main

m = Main()

#c3s_vars = ['al_bbdh','al_bbbh','al_spdh','al_spbh','lai','fapar']
c3s_vars = ['al_bbdh']
c3s_vars = ['c3s_'+i for i in c3s_vars]
#c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR','_VGT','_PROBAV','_SENTINEL3']]
c3s_vars = [i+r for i in c3s_vars for r in ['_AVHRR']]
#c3s_vars = [i+'_AVHRR '+i+'_VGT' for i in c3s_vars]

for v in c3s_vars:
    #args = "-t0 1981-09-01 -t1 1981-09-30 -i latloncsv:config -p {} -a extract -d --debug 1 --config config_vito.yml".format(v).split()
    #args = "-t0 1981-09-01 -t1 1984-12-31 -i latloncsv:config -p {} -a extract merge trend plot --debug 1 --config config_vito.yml".format(v).split()
    args = "-t0 1981-09-01 -t1 2007-12-31 -i latloncsv:config -p {} -a extract merge trend plot --config config_vito.yml".format(v).split()
    #args = "-t0 1981-09-01 -t1 1981-12-31 -i latloncsv:config -p {} -a extract merge trend --config config_vito.yml".format(v).split()
    #args = "-t0 1981-01-01 -t1 1981-12-31 -i latloncsv:config -p {} -a extract merge trend plot -d --debug 1 --config config_vito.yml".format(v).split()
    #args = "-t0 1998-03-01 -t1 1998-05-31 -i latloncsv:config -p {} -a extract merge trend plot -d --debug 1 --config config_vito.yml".format(v).split()
    m.preprocess(args)
    m.process()

