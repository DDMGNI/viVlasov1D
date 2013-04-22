#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=72:00:00
#
#$ -pe mpich2_tok_production 8
#
#$ -o /ptmp1/mkraus/petscVlasovPoisson1D/landau01_201x401_dt1E-2_nu0.out
#$ -e /ptmp1/mkraus/petscVlasovPoisson1D/landau01_201x401_dt1E-2_nu0.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=landau01_201x401_dt1E-2_nu0


module load intel/13.1
module load mkl/11.0
module load impi/4.1.0

module load python32/all

export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


mpiexec -n 8 python3.2 run_direct_nonlinear_exact.py runs/$RUNID.cfg
