import os
from pathlib import Path

import optax
from dopamine.discrete_domains import gym_lib
from dopamine.jax import losses

from thesis import exploration, networks
from thesis.memory import offline_memory
from thesis.reporter import reporter

base_dir = Path(os.path.dirname(__file__))
dopamine_dir = Path(base_dir.parent, "dopamine")

data_dir = Path(base_dir.parent.parent, "resources", "data")
aim_dir = data_dir
scratch_data_dir = data_dir.joinpath("scratch")
peregrine_data_dir = "/data/s3680622"

scratch_dir = base_dir.joinpath("scratch")


make_batch_rl_agent = lambda agent_class: {
    "call_": agent_class,
    "net_sync_freq": int(2e3),
    "train_freq": 4,
    "clip_rewards": True,
    "min_replay_history": int(2e3),
}

make_env = lambda env, version, creator=gym_lib.create_gym_environment: {
    "call_": creator,
    "environment_name": env,
    "version": version,
}

make_reporters = lambda mongo_collection_name, mongo_buffering=50, aim_repo=str(
    data_dir
): {
    "mongo": {
        "call_": reporter.MongoReporter,
        "buffering": mongo_buffering,
        "collection_name": mongo_collection_name,
    },
    "aim": {"call_": reporter.AimReporter, "repo": aim_repo},
}

make_batch_rl_memory = lambda parallel=True: {
    "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
    "batch_size": 256,
    "load_parallel": parallel,
}

adam_huberloss = {
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss_metric": losses.huber_loss,
}

classic_control_mlp_huberloss_adam = {
    "model": {"hiddens": (512, 512)},
    **adam_huberloss,
}

convnet_huberloss_adam = {
    "model": {"call_": networks.NatureDQNNetwork},
    **adam_huberloss,
}

egreedy_exploration = {"call_": exploration.egreedy, "epsilon_eval": 0.001}
