import itertools
import os
import subprocess
import uuid
import time


n_output_neurons = 10
gamma = [5e-7, 1e-6, 5e-6]
tau = [1e-3, 5e-3, 10e-3]
lower_freq = [5, 10, 20]
upper_freq = [75, 100, 200]
image_threshold = [50, 200]

combinations = list(
    itertools.product(gamma, tau, lower_freq, upper_freq, image_threshold)
)

cwd = os.getcwd()
for combination in combinations:
    d = {
        "gamma": combination[0],
        "tau": combination[1],
        "lower_freq": combination[2],
        "upper_freq": combination[3],
        "image_threshold": combination[4],
    }
    args = ""
    for key in d:
        args = args + ' --' + key + ' ' + str(d[key])

    bash_script = """#!/bin/bash
#PBS -j oe
#PBS -m ae
#PBS -N SNN-GrapheneSynapses-%d
#PBS -o "%s"
#PBS -e "%s"
#PBS -M corey.lammie@jcu.edu.au
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=10gb
shopt -s expand_aliases
source /etc/profile.d/modules.sh
cd "%s"
. ~/.bash_profile
module load conda3
source $CONDA_PROF/conda.sh
conda activate base
python Grid.py%s
""" % (n_output_neurons, os.path.join(cwd, 'out.txt'), os.path.join(cwd, 'out.txt'), cwd, args)
    
    myuuid = str(uuid.uuid4())
    with open(os.path.join(os.getcwd(), "%s.sh" % myuuid), "w+") as f:
        f.writelines(bash_script)

    res = subprocess.run("qsub %s.sh" % myuuid, capture_output=True, shell=True)
    print(args)
    print(res.stdout.decode())
    os.remove(os.path.join(os.getcwd(), "%s.sh" % myuuid))
    time.sleep(2)


