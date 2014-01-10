#!/bin/bash
#
#$ -cwd
#
#$ -l h_cpu=00:30:00
#
#$ -pe mpich2_tok_production 16
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=benchmark_16_201x401


module load intel/12.1
module load mkl/11.1
module load impi/4.1.0

module load python33/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_core.so:/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_intel_thread.so:/afs/@cell/common/soft/intel/ics13/13.0/compiler/lib/intel64/libiomp5.so


mpiexec -n 16 python3.2 run_direct_nonlinear.py runs_itm/$RUNID.cfg


