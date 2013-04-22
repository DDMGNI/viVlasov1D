#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=96:00:00
#
#$ -pe mpich2_tok_production 8
#
#$ -o /ptmp1/mkraus/petscVlasovPoisson1D/jeans_strong_201x401_dt1E-1_nu4E-4.out
#$ -e /ptmp1/mkraus/petscVlasovPoisson1D/jeans_strong_201x401_dt1E-1_nu4E-4.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=jeans_strong_201x401_dt1E-1_nu4E-4


module purge

export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/compilers:${MODULEPATH}
export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/libs:${MODULEPATH}
export MODULEPATH=/afs/ipp/common/usr/modules/@sys/modulefiles/TOK:${MODULEPATH}

module load hdf5-serial/1.8.9
module load netcdf-serial

module load intel/13.1
module load mkl/11.0
module load impi/4.1.0
module load llvm
module load cmake

module load python32/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_newton.py runs/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear.py runs/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_corr.py runs/$RUNID.cfg
mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_ksp.py runs/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_pred.py runs/$RUNID.cfg