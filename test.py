#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main import Main


def run_test(n):
    m = Main()

    ### List all tests

    test_args = []

    # Quick run to check the extraction (5 points extract, merge)
    test_args.append("-t0 1981-09-01 -t1 1981-09-30 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge -f --debug 1 --config config_vito.yml".split())
    
    # Quick run to check the pipeline (previous + plot)
    test_args.append("-t0 1981-01-01 -t1 1983-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -f --debug 1 --config config_vito.yml".split())

    # Quick run to check the pipeline (previous + all points)
    test_args.append("-t0 1981-01-01 -t1 1983-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -f --config config_vito.yml".split())

    # S3 with 5 points
    test_args.append("-t0 2018-07-01 -t1 2018-07-31 -i latloncsv:config -p c3s_al_bbdh_SENTINEL3 -a extract --debug 1 --config config_vito.yml".split())
    # S3 with all points
    test_args.append("-t0 2018-07-01 -t1 2018-07-31 -i latloncsv:config -p c3s_al_bbdh_SENTINEL3 -a extract --config config_vito.yml".split())

    # Long test
    test_args.append("-t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot --config config_vito.yml".split())
    


    ### Run test
    print("Tested command:")
    print("python main.py", " ".join(test_args[n]))
    m.preprocess(test_args[n])
    m.process()
    return 0


def test_AVHRR_merge_5():
    assert run_test(0)==0

def test_AVHRR_plot_5():
    assert run_test(1)==0

def test_AVHRR_plot_all():
    assert run_test(2)==0

def test_S3_extract_5():
    assert run_test(3)==0

def test_S3_extract_all():
    assert run_test(4)==0

def test_long_albbdh_plot_all():
    assert run_test(5)==0


