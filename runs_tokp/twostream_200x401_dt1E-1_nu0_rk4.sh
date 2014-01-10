#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=24:00:00
#
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


RUNID=twostream_201x400_dt1E-1_nu0_rk4


module load hdf5-serial/1.8.9
module load netcdf-serial/4.2.1.1

module load intel/12.1
module load mkl/11.1
module load impi/4.1.0
module load llvm
module load cmake

module load python33/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics/2011.0.013/12.1/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD

#export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
#export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
#export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


mpiexec -perhost 16 -l -n 16 python3.3 run_nonlinear_matrixfree_split_rk4.py -c runs_itm/$RUNID.cfg
