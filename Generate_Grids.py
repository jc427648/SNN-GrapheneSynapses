import itertools
import os
import subprocess
from subprocess import PIPE
import uuid
import time
import numpy as np


n_output_neurons = 10
iterations = 10
tau = [4.8025530126437]
gamma = [0.00002660854454112]
# C2CD2D = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
C2CD2D = np.append(np.arange(0., 0.05 + 0.001, step=0.001), [0.1])

combinations = list(
    itertools.product(tau, gamma, C2CD2D)
)
print(len(combinations) * iterations)

cwd = os.getcwd()
for combination in combinations:
    for iteration in range(iterations):
        myuuid = str(uuid.uuid4())
        d = {
            "n_output_neurons": n_output_neurons,
            "tau": combination[0],
            "gamma": combination[1],
            "C2CD2D": combination[2],
            "UUID": myuuid,
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
        with open(os.path.join(os.getcwd(), "%s.sh" % myuuid), "w+") as f:
            f.writelines(bash_script)

        res = subprocess.run("qsub %s.sh" % myuuid, stdout=PIPE, stderr=PIPE, shell=True)
        print(args)
        print(res.stdout.decode('utf-8'))
        os.remove(os.path.join(os.getcwd(), "%s.sh" % myuuid))
        time.sleep(1)