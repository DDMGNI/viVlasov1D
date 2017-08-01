#!/bin/bash
#
#$ -cwd
#
#$ -l h_rt=24:00:00
#
#$ -P  tokp
#$ -pe impi_hydra 16
#
#$ -o /tokp/scratch/mkraus/viVlasov1D/jeans_strong_1024x8192.$JOB_ID.out
#$ -e /tokp/scratch/mkraus/viVlasov1D/jeans_strong_1024x8192.$JOB_ID.err
#
#$ -m beas
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=jeans_strong_1024x8192


module load intel/14.0
module load mkl/11.1
module load impi/4.1.3
module load fftw/3.3.3
module load hdf5-mpi/1.8.12

export FFTW_HOME=/afs/@cell/common/soft/fftw/fftw-3.3.3/@sys/intel-13.1/impi-4.1

module load py33-python
module load py33-cython
module load py33-numpy
module load py33-scipy
module load py33-configobj
module load py33-pyfftw
module load py33-mpi4py
module load py33-petsc4py


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


mpiexec -perhost 16 -l -n 16 python3.3 run.py -c runs_tokp/$RUNID.cfg -i $JOB_ID