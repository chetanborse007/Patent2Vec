#!/bin/sh

#PBS -q copperhead
#PBS -N Patent2VecApp
#PBS -l nodes=1:ppn=8,mem=512gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR

python $PWD/Patent2VecApp.py
