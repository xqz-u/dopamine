#!/bin/bash

# NOTE source this script baby

# echo "Cloning xqz-u/dopamine..."
# git clone https://github.com/xqz-u/dopamine.git
xqzu_thesis_dir=$(pwd)
xqzu_thesis_home=$xqzu_thesis_dir/dopamine
xqzu_thesis_data=$xqzu_thesis_dir/resources/data
echo "Thesis python root: $xqzu_thesis_home"
echo "Thesis data folder: $xqzu_thesis_data"

echo "Installing poetry..."
curl -sSL https://install.python-poetry.org | python3 -
echo "poetry stuff: version $(poetry --version) bin: $(which poetry)"

cd $xqzu_thesis_home
echo "Create poetry env and install dependencies..."
poetry install

echo "Source poetry environment..."
# poetry shell
source $(poetry env info --path)/bin/activate
echo "Using python: $(which python) version: $(python --version)"

echo "Downloading Pong DQN replay data (~12.2GB) to $xqzu_thesis_data"
gsutil -m cp -R gs://atari-replay-datasets/dqn/Pong $xqzu_thesis_data

echo "Starting MongoDB Docker container..."
cd $xqzu_thesis_data/mongodb
sudo docker-compose up -d

cd $xqzu_thesis_home
echo "Train DQVMax offline on Pong!"
time python -m thesis.experiments.dqvmax_pong_offline


# uninstall poetry
# curl -sSL https://install.python-poetry.org | python3 - --uninstall
