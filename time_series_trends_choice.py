#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:41:51 2020

@author: moparthys
"""

import numpy as np
import time_series_trends
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='The parameters are being given as arguments for input time series,', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-styr','--start_year', help='start year for read of time series')    
    parser.add_argument('-enyr','--end_year', help='end year for read of time series')    
    parser.add_argument('-stmn','--start_month', help='start month for read of time series')    
    parser.add_argument('-enmn','--end_month', help='end month for read of time series')    
    parser.add_argument('-o','--output', help='output path')    
    parser.add_argument('-ptag','--product_tag', help='product tag or product dataset, either albedo, lai, evapo, dssf, fapar ')    
    parser.add_argument('-x1','--xlim1', help='limit x1 ')    
    parser.add_argument('-x2','--xlim2', help='limit x2 ')    
    parser.add_argument('-y1','--ylim1', help='limit y1 ')    
    parser.add_argument('-y2','--ylim2', help='limit y2 ')  
    parser.add_argument('-n_master','--master_chunk', help='size of master chunks')
    
    parser.add_argument('-c','--choice', help='product tag or product dataset, either ALBEDO, LAI, EVAPO, DSSF, FAPAR ')    

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    start_year=int(args.start_year)
    end_year=int(args.end_year)
    
    start_month=int(args.start_month)
    end_month=int(args.end_month)
    
    output_path=args.output
    
    product_tag=args.product_tag
    
    xlim1=int(args.xlim1)
    xlim2=int(args.xlim2)
    ylim1=int(args.ylim1)
    ylim2=int(args.ylim2)
    nmaster=int(args.master_chunk)
        
    choice=args.choice
    
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
