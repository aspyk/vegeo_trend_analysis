#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main import Main


def run_test(n):
    m = Main()

    ### List all tests

    test_args = []

    # Quick run to check the pipeline (5 points extract, merge, trends)
    test_args.append("-t0 1981-09-01 -t1 1981-09-30 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend -d --debug 1 --config config_vito.yml".split())
    
    # Quick run to check the pipeline (previous + plot)
    test_args.append("-t0 1981-01-01 -t1 1981-10-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -d --debug 1 --config config_vito.yml".split())

    # Quick run to check the pipeline (previous + all points)
    test_args.append("-t0 1981-01-01 -t1 1981-10-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -d --config config_vito.yml".split())

    # S3 with 5 points
    test_args.append("-t0 2018-07-01 -t1 2018-07-31 -i latloncsv:config -p c3s_al_bbdh_SENTINEL3 -a extract --debug 1 --config config_vito.yml".split())
    # S3 with all points
    test_args.append("-t0 2018-07-01 -t1 2018-07-31 -i latloncsv:config -p c3s_al_bbdh_SENTINEL3 -a extract --config config_vito.yml".split())


    ### Run test

    m.preprocess(test_args[n])
    m.process()
    return 0


def test1():
    assert run_test(0)==0

def test2():
    assert run_test(1)==0

def test3():
    assert run_test(2)==0

def test4():
    assert run_test(3)==0

def test5():
    assert run_test(4)==0


