#! /usr/bin/bash

# TODO mongod config:
# - increase snapshot time (~2m rn?)

# NOTE the singularity image should be present on the host; it can
# be built locally with:
# sudo singularity build mongo_latest.sif ../resources/data/mongodb/Singularity

# NOTE very important: close any running mongodb instance on the
# interactive node before submitting a job! since they use the same db
# - same files - only one can run at a time. so either start the db on
# a compute node and connect to it at localhost:27017, or find a way to
# connect to the interactive node from the compute node

cd /data/$USER
# to install the setup:
# echo 'export SINGULARITY_CACHEDIR=/data/$USER/.singularity' >> ~/.bashrc
# mkdir mongodb mongodb_logs

singularity instance start \
	    -B mongodb:/data/db \
	    -B $HOME/thesis/resources/data/mongodb/mongoconf:/etc/mongo/ \
	    -B mongodb_logs:/var/log/mongodb \
	    mongo_latest.sif mongodb
