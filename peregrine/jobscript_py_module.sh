#!/bin/bash

# NOTE cl switches:
# -J, --job-name
# -o, --output
# -c, --cpus-per-task
# example call: sbatch --mem 3G -t 05:00:00 -J <jobname> -c 6 jobscript_py_module.sh ...

# NOTE next directive is for job profiling and viz on Grafana
#SBATCH --profile=task
### SBATCH --partition=gpu
### SBATCH --gres=gpu:1


PG_SCRIPTS=$HOME/thesis/peregrine


module purge
source $PG_SCRIPTS/activate_thesis_env.sh

# give some time to container to startup
$PG_SCRIPTS/start_mongo.sh && sleep 5

cd $HOME/thesis/dopamine
echo "Run python module: $1"
python -m $1

# close any running mongo container on the allocated compute node
$PG_SCRIPTS/stop_mongo.sh


# results for single iteration train/eval:
# -classic control (DQVMax, CartPole):
# ~27/30s on my pc and pg interactive gpu node / ~1.5 local eval
# ~0.027 on pg gpu job, x1000 speedup ?!
