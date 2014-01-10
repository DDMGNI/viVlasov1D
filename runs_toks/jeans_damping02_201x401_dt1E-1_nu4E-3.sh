#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=96:00:00
#
#$ -pe mpich2_tok_production 8
#
#$ -o /ptmp1/mkraus/petscVlasovPoisson1D/jeans_damping02_201x401_dt1E-1_nu4E-3.out
#$ -e /ptmp1/mkraus/petscVlasovPoisson1D/jeans_damping02_201x401_dt1E-1_nu4E-3.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=jeans_damping02_201x401_dt1E-1_nu4E-3


module load intel/13.1
module load mkl/11.0
module load impi/4.1.0

module load python32/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_core.so:/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_intel_thread.so:/afs/@cell/common/soft/intel/ics13/13.0/compiler/lib/intel64/libiomp5.so


mpiexec -n 8 python3.2 run_direct_nonlinear.py runs_tokbc/$RUNID.cfg
