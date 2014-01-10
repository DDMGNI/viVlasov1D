#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=168:00:00
#
#$ -pe impi_hydra 16
#
#$ -o /ptmp1/mkraus/petscVlasovPoisson1D/bump_on_tail_201x401_dt1E-1_nu1E-5.out
#$ -e /ptmp1/mkraus/petscVlasovPoisson1D/bump_on_tail_201x401_dt1E-1_nu1E-5.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=bump_on_tail_201x401_dt1E-1_nu1E-5


module load intel/13.1
module load mkl/11.0
module load impi/4.1.0

module load python33/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_core.so:/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_intel_thread.so:/afs/@cell/common/soft/intel/ics13/13.0/compiler/lib/intel64/libiomp5.so


mpiexec -n 8 python3.2 run_direct_nonlinear.py runs_itm/$RUNID.cfg
