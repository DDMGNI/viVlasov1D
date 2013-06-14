#
#$ -cwd
#
#$ -l h_rt=24:00:00
#
#$ -pe impi_hydra 16
#
#$ -o /pfs/scratch/mkraus/petscVlasovPoisson1D/jeans_strong_200x401_dt1E-1_nu1E-3.out
#$ -e /pfs/scratch/mkraus/petscVlasovPoisson1D/jeans_strong_200x401_dt1E-1_nu1E-3.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viVlasov1D
#


RUNID=jeans_strong_200x401_dt1E-1_nu1E-3


module purge

export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/compilers:${MODULEPATH}
export MODULEPATH=/afs/@cell/common/usr/modules/@sys/modulefiles/libs:${MODULEPATH}
export MODULEPATH=/afs/ipp/common/usr/modules/@sys/modulefiles/TOK:${MODULEPATH}

module load hdf5-serial/1.8.9
module load netcdf-serial/4.2.1.1

module load intel/13.1
module load mkl/11.0
module load impi/4.1.0
module load llvm
module load cmake

module load python32/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.1/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nested.py runs_itm/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_newton.py runs_itm/$RUNID.cfg
mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear.py runs_itm/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_corr.py runs_itm/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_ksp.py runs_itm/$RUNID.cfg
#mpiexec -perhost 16 -l -n 16 python3.2 run_direct_nonlinear_pred.py runs_itm/$RUNID.cfg

