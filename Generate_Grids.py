import itertools
import os
import subprocess
import uuid
import time

#You can run this by going dummy.py, but I don't think this code should take too long to run.

n_output_neurons = 100
# gamma = [0.8e-6, 1e-6, 1.2e-6]
tau = [2.3,2.4,2.5,2.6,2.7,2.8]
# R = [900,1000,1100,800]
# lower_freq = [10, 20, 30]
# upper_freq = [75, 100,125]
# image_threshold = [1,2,5]
# target_activity = [1]
n_epochs = [1,2,3]
# image_duration = [0.03,0.04,0.045,0.055,0.06,0.07,0.08,0.09,0.1]

combinations = list(
    itertools.product(n_epochs,tau)
)

cwd = os.getcwd()
for combination in combinations:
    d = {
        # "gamma": combination[0],
        "tau": combination[1],
        # "image_duration": combination[0],
        # "target_activity": combination[1],
        # "n_epochs": combination[0],
        # "lower_freq": combination[1],
        # "upper_freq": combination[2],
        #"image_threshold": combination[4],
        "n_epochs": combination[0],
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
#PBS -M ben.walters@jcu.edu.au
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=10gb
shopt -s expand_aliases
source /etc/profile.d/modules.sh
cd "%s"
. ~/.bash_profile
module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/benwalters/conda_env
cd /scratch/user/benwalters/SNN-GrapheneSynapses
python Grid.py%s
""" % (n_output_neurons, os.path.join(cwd, 'out.txt'), os.path.join(cwd, 'error.txt'), cwd, args)
    
    myuuid = str(uuid.uuid4())
    with open(os.path.join(os.getcwd(), "%s.sh" % myuuid), "w+") as f:
        f.writelines(bash_script)

    res = subprocess.run("qsub %s.sh" % myuuid, capture_output=True, shell=True)
    print(args)
    print(res.stdout.decode())
    os.remove(os.path.join(os.getcwd(), "%s.sh" % myuuid))
    time.sleep(2)