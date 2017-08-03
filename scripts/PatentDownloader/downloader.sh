#!/bin/sh

#PBS -q copperhead
#PBS -N PatentDownloader
#PBS -l nodes=1:ppn=16,mem=768gb
#PBS -l walltime=25:00:00

cd $PBS_O_WORKDIR

python $PWD/PatentDownloader.py
