#! /usr/bin/bash

module load jq
tail -f /data/$USER/mongodb_logs/mongod.log | jq
