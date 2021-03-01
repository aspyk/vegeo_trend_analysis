f2py -c mk_fortran.f90 -m mankendall_fortran_repeat_exp2 --fcompiler=gfortran
#python main.py -t0 1981-01-01 -t1 1981-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR  -a extract merge trend plot -d
python main.py -t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot -d
