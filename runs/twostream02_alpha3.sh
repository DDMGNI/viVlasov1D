#!/bin/bash
#
#$ -cwd
#
#$ -j y
#
#$ -l h_cpu=24:00:00
#
#$ -pe mpich2_tok_production 8
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N petscVlasov1D
#


RUNID=twostream02_alpha3


module load intel/13.0
module load mkl/11.0
module load impi/4.1.0

module load python32/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_core.so:/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_intel_thread.so:/afs/@cell/common/soft/intel/ics13/13.0/compiler/lib/intel64/libiomp5.so


mpiexec -n 8 python3.2 run_direct_nonlinear.py runs/$RUNID.cfg
