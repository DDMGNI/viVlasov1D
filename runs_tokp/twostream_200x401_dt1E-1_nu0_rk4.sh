#!/bin/bash
#
#$ -cwd
#
#$ -l h_rt=24:00:00
#
#$ -P  tokp
#$ -pe impi_hydra 16
#
#$ -o /tokp/scratch/mkraus/petscVlasovPoisson1D/twostream_200x401_dt1E-1_nu0_rk4.out
#$ -e /tokp/scratch/mkraus/petscVlasovPoisson1D/twostream_200x401_dt1E-1_nu0_rk4.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=twostream_200x401_dt1E-1_nu0_rk4


module load intel/14.0
module load mkl/11.1
module load impi/4.1.0
module load fftw/3.3.3
module load hdf5-mpi/1.8.9

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

mpiexec -perhost 16 -l -n 16 python3.3 run_nonlinear_matrixfree_split_rk4.py -c runs_tokp/$RUNID.cfg
