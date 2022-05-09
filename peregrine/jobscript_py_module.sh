#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --ntasks=6
#SBATCH --mem=3GB
# NOTE next directive is for job profiling and viz on Grafana
#SBATCH --profile=task
### SBATCH --partition=gpu
### SBATCH --gres=gpu:1


# NOTE cl switches:
# -J, --job-name
# -o, --output


PG_SCRIPTS=$HOME/thesis/peregrine


module purge
source $PG_SCRIPTS/activate_thesis_env.sh

# close any running mongo container
$PG_SCRIPTS/stop_mongo.sh
# give some time to container to startup
$PG_SCRIPTS/start_mongo.sh && sleep 5

cd $HOME/thesis/dopamine
echo "Run python module: $1"
python -m $1


# results
# -classic control (DQVMax, CartPole):
# ~27/30s on my pc and pg interactive gpu node
# ~0.027 on pg gpu job, x1000 speedup ?!
