
import os

from vlasov.core.config  import Config


# test cases
run_ids = [
        ("landau_linear", "Linear Landau Damping"),
        ("landau_nonlinear", "Nonlinear Landau Damping"),
        ("twostream", "Twostream Instability"),
        ("twostream_shifted", "Twostream Instability (shifted)"),
        ("bump_on_tail", "Bump-on-tail"),
        ("jeans_damping", "Jeans Damping"),
        ("jeans_weak", "Weak Jeans Instability"),
        ("jeans_strong", "Strong Jeans Instability"),
    ]

# grid resolution (nx,nv)
resolutions = [
        ( 64, 128),
        (128, 256),
        (256, 512),
        (512,1024),
    ]

# collision frequencies
collisions = [
        0E+0,
        1E-5,
        1E-4,
        2E-4,
        4E-4,
    ]

# run file directory
run_dir = "runs_tokp"

# run script filename
run_filename = "submit_paper_runs.sh"

# run script header
run_script = """
#!/bin/bash
"""

# output directory
out_dir = run_dir + "/paper/"

# create output directory
try:
    os.stat(out_dir)
except:
    os.mkdir(out_dir)


# loop over all test cases
for (run_id, run_title) in run_ids:
    # read template config file
    cfg = Config(run_dir + "/" + run_id + "/" + run_id + ".cfg")
    
    # add section to run script
    run_script += "\n"
    run_script += "# " + run_title + "\n"
    run_script += "\n"
    
    # loop over all grid resolutions
    for (nx,nv) in resolutions:
        # loop over all collision frequencies
        for nu in collisions:
            
            # set grid resolution
            cfg['grid']['nx'] = nx
            cfg['grid']['nv'] = nv
            
            # set collision frequency
            cfg['solver']['coll_freq'] = nu
            
            # compose run name
            run_name = run_id \
                     + "_" + str(nx) + "x" + str(nv) \
                     + "_nu%1.0E" % nu
            
            # compose config file name
            outfile = out_dir + "/" +  run_name + ".cfg"
            
            # create config files for each run
            cfg.write_current_config(outfile)
            
            # add entry in run script
            run_script += "qsub -N " + run_name + " runs_tokp/run_paper.sh"
            run_script += "\n"
        
        run_script += "\n"
            

# save run script
f = open(run_filename, "w")
f.write(run_script)
f.close()

# print run script
print(run_script)
