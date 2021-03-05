#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os,sys

from main import Main

m = Main()

for i in [0]:
    args = "-t0 1981-01-01 -t1 1981-10-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -d --debug 1 --config config_vito.yml".split()
    m.preprocess(args)
    m.process()

