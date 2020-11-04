#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import datetime
import os,sys



def parse_args():
    parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t0','--start_date', help='start date (ISO format %%Y-%%m-%%d) for reading of time series.', type=lambda s: datetime.date.fromisoformat(s))    
    parser.add_argument('-t1','--end_date', help='end date (ISO format %%Y-%%m-%%d) for reading of time series.', type=lambda s: datetime.date.fromisoformat(s))    
    parser.add_argument('-o','--output', help='output path')    
    parser.add_argument('-ptag','--product_tag', help='product tag or product dataset, either albedo, lai, evapo, dssf, fapar ')    
    parser.add_argument('-zc','--zone_coor', help='Zone to be analysed. Given as <xmin xmax ymin ymax> as the boundary box coordinates.', nargs='+', type=int) 
    parser.add_argument('-zn','--zone_name', help='Zone to be analysed. Given as a name, either Euro, NAfr, SAfr, SAme.')    
    parser.add_argument('-n_master','--master_chunk', help='size of master chunks', type=int)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    start_year = int(args.start_date.year)
    end_year = int(args.end_date.year)
    
    start_month = int(args.start_date.month)
    end_month = int(args.end_date.month)
    
    output_path = args.output
    
    product_tag = args.product_tag
    
    print(args.zone_coor)
    print(args.zone_name)

    xlim1 = args.zone_coor[0] 
    xlim2 = args.zone_coor[1]
    ylim1 = args.zone_coor[2]
    ylim2 = args.zone_coor[3]
    
    nmaster = args.master_chunk
        
   
    dic_zone = {}
    dic_zone['Euro'] = [1550, 3250, 50, 700]
    dic_zone['NAfr'] = [1240, 3450, 700, 1850]
    dic_zone['SAfr'] = [2140, 3350, 1850, 3040]
    dic_zone['SAme'] = [40, 740, 1460, 2970]


    sys.exit()

    import time_series_trends


    import estimate_trends_from_time_series

    import trend_file_merger
    if choice=="ALBEDO":
        time_series_trends.time_series_albedo(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)
    if choice=="LAI":
        time_series_trends.time_series_lai(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)
    if choice=="FAPAR":
        time_series_trends.time_series_fapar(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)
    if choice=="EVAPO":
        time_series_trends.time_series_evapo(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)
    if choice=="DSSF":
        time_series_trends.time_series_dssf(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)
    if choice=="LST":
        time_series_trends.time_series_lst(start_year,end_year,start_month,end_month,output_path,product_tag,xlim1,xlim2,ylim1,ylim2,nmaster)


if __name__ == "__main__":
   main()

