#! /usr/bin/bash

# TODO pass a config to mongo singularity container:
# - increase snapshot time (2m rn?)

cd /data/$USER

# NOTE to install the setup:
# echo 'export SINGULARITY_CACHEDIR=/data/$USER/.singularity' >> ~/.bashrc
# singularity pull docker://mongo
# mkdir mongodb mongodb_logs

# singularity run --bind mongodb:/data/db mongo_latest.sif # foreground
singularity instance start \
	    -B mongodb:/data/db \
	    -B $HOME/thesis/resources/data/mongodb/mongoconf:/etc/mongo/ \
	    -B mongodb_logs:/var/log/mongodb \
	    mongo_latest.sif mongodb
singularity run instance://mongodb
