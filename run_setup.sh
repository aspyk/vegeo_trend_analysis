#!/bin/bash

# Change branch
git checkout feature/genericreader

# Fortran compilation
f2py -c mk_fortran.f90 -m mankendall_fortran_repeat_exp2 --fcompiler=gfortran

