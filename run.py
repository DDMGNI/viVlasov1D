#!/usr/bin/env python
'''
Created on Jun 26, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, sys
import pstats, cProfile

from petsc4py import PETSc

from vlasov.core.config  import Config


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('-c', '--config', metavar='<cfg_file>', type=str, required=True,
                        help='Configuration File')
    parser.add_argument('-i', '--runid',  metavar='<runid>',    type=str, required=False, default=None,
                        help='Run ID')
    parser.add_argument('-ksp', '--kspmonitor',  action='store_true', required=False,
                        help='Show PETSc KSP monitor')
    parser.add_argument('-snes','--snesmonitor', action='store_true', required=False,
                        help='Show PETSc SNES monitor')
    parser.add_argument('-info','--loginfo', action='store_true', required=False,
                        help='Show PETSc info')
    parser.add_argument('-summary','--logsummary', action='store_true', required=False,
                        help='Show PETSc summary')
    parser.add_argument('-prof','--profiler', action='store_true', required=False,
                        help='Activate Profiler')
    
    args = parser.parse_args()
    
    
    # set some PETSc options
    OptDB = PETSc.Options()
    
    if args.kspmonitor:
        OptDB.setValue('ksp_monitor',  '')
    
    if args.snesmonitor:
        OptDB.setValue('snes_monitor', '')
    
    if args.loginfo:
        OptDB.setValue('log_info',  '')
    
    if args.logsummary:
        OptDB.setValue('log_summary',  '')
    
#     OptDB.setValue('snes_ls', 'basic')

    
    cfg = Config(args.config)
    
    # determine runscript to load
    # the structure is like
    # run[_<type>]_<method>[_<initial_guess>][_<mode>][_<timestepping>][_pc_<preconditioner_type>]
    
    run_script = "run"
    
    if cfg['solver']['method'] != 'explicit':
        run_script += "_" + cfg['solver']['type']
    
    run_script += "_" + cfg['solver']['method'] 
    
    if cfg['solver']['method'] == 'explicit':
        run_script += "_" + cfg['solver']['initial_guess'] 
    else:
        run_script += "_" + cfg['solver']['mode'] 
        run_script += "_" + cfg['solver']['timestepping'] 
    
        if cfg['solver']['preconditioner_type'] != None:
            if cfg['solver']['preconditioner_scheme'] != None:
                run_script += "_pc" 
                run_script += "_" + cfg['solver']['preconditioner_type']
            else:  
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("ERROR: preconditioner type is set to %s, but preconditioner scheme is not set." % cfg['solver']['preconditioner_type'])
                sys.exit()
            
    
    if PETSc.COMM_WORLD.getRank() == 0:
        print("")
        print("Loading runscript      %s" % (run_script))
    
    
    run_object = __import__(run_script, globals(), locals(), ['petscVP1Drunscript'], 0)
    
    with run_object.petscVP1Drunscript(cfg=cfg, runid=args.runid) as petscvp:
        if args.profiler:
            cProfile.runctx("petscvp.run()", globals(), locals(), "profile.prof")
        else:
            petscvp.run()
    
    if args.profiler:
        s = pstats.Stats("profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    
