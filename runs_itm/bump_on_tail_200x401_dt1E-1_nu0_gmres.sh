#
#$ -cwd
#
#$ -l h_rt=24:00:00
#
#$ -pe impi_hydra 16
#
#$ -o /pfs/scratch/mkraus/petscVlasovPoisson1D/bump_on_tail_200x401_dt1E-1_nu0_gmres.out
#$ -e /pfs/scratch/mkraus/petscVlasovPoisson1D/bump_on_tail_200x401_dt1E-1_nu0_gmres.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=bump_on_tail_200x401_dt1E-1_nu0_gmres


module purge

export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/compilers:${MODULEPATH}
export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/libs:${MODULEPATH}
export MODULEPATH=/afs/ipp/common/usr/modules/@sys/modulefiles/TOK:${MODULEPATH}


module load hdf5-serial/1.8.9
module load netcdf-serial/4.2.1.1

#module load intel/13.1
module load mkl/11.0
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


mpiexec -perhost 16 -l -n 16 python3.3 run_nonlinear_matrixfree_split.py -c runs_itm/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.3 run_nonlinear_matrixfree_split_pc_exp.py runs_itm/$RUNID.cfg

