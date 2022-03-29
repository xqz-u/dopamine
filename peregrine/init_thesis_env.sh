#! /usr/bin/sh

module load Python/3.9.6-GCCcore-11.2.0

curl -sSL https://install.python-poetry.org | python3 -

git clone https://github.com/xqz-u/dopamine.git

cd dopamine

poetry config cache-dir /data/s3680622/.cache/pypoetry
poetry install
