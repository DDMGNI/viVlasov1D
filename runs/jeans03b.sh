#!/bin/bash
#
#$ -cwd
#
#$ -j y
#
#$ -l h_cpu=48:00:00
#
#$ -pe mpich2_tok_devel 8
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N pyVlasov1D
#


RUNID=jeans03b


module load intel/12.1
module load mkl/10.3
module load hdf5-serial
module load python27/python
module load python27/numpy
module load python27/scipy
module load python27/cython
module load python27/h5py

export OMP_NUM_THREADS=8

export PYTHONPATH=/afs/ipp/home/m/mkraus/Python/code/pyVlasov1D:$PYTHONPATH


python pyvp1d.py runs/$RUNID.cfg
