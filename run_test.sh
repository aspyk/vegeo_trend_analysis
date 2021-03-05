#!/bin/bash

## Quick run to check the pipeline (5 points extract, merge, trends)
#python main.py -t0 1981-09-01 -t1 1981-09-30 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend -d --debug 1 --config config_vito.yml

## Same with plot
python main.py -t0 1981-01-01 -t1 1981-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR -a extract merge trend plot -d --debug 1 --config config_vito.yml
#python main.py -t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot -d
### Test SENTINEL3
#python main.py -t0 2018-07-01 -t1 2018-07-31 -i latloncsv:config -p c3s_al_bbdh_SENTINEL3 -a extract --debug 1 --config config_vito.yml
