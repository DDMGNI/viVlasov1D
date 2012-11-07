#!/bin/bash
#
#$ -cwd
#
#$ -j y
#
#$ -l h_cpu=24:00:00
#
#$ -pe mpich2_tok_devel 32
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N petscVlasovPoisson1D
#


RUNID=landau02e1


module load intel/12.1
module load mkl/10.3
module load impi
module load hdf5-serial

module load python27/python
module load python27/numpy
module load python27/h5py
module load python27/mpi4py
module load python27/petsc4py
module load python27/cython


export RUN_DIR=/afs/ipp/home/m/mkraus/Codes/petscVlasovPoisson1D

export PYTHONPATH=$RUN_DIR:$PYTHONPATH


cd $RUN_DIR

mpiexec -np 32 python petscvp1d.py runs/$RUNID.cfg
