import logging
import os

import optax
from dopamine.jax import losses
from thesis import config, offline_circular_replay_buffer
from thesis.agents import DQNAgent
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner, runner

make_config = lambda exp_name, env, version: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": {
            "model": {"hiddens": (512, 512)},
            "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
            "loss_metric": losses.huber_loss,
        }
    },
    "exploration": {},
    "agent": {
        "call_": DQNAgent.DQNAgent,
        "net_sync_freq": int(1e4),
        # "min_replay_history": int(5e3),
        "min_replay_history": 500,
    },
    # "memory": {"replay_capacity": int(5e4)},
    "memory": {"replay_capacity": 700},
    "env": {"environment_name": env, "version": version},
    "runner": {
        "log_level": logging.DEBUG,
        "experiment": {
            "schedule": "train",
            # "record_experience": True,
            "seed": 4,
            "steps": 1000,
            "iterations": 3,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 25,
            "collection_name": exp_name,
        },
        "aim": {"call_": reporter.AimReporter, "repo": str(config.data_dir)},
    },
}


online_spec = [
    ("test_cp_record", "CartPole", "v1"),
    ("test_ab_record", "Acrobot", "v1"),
]
online_confs = []
for name, env, v in online_spec:
    c = make_config(name, env, v)
    c["runner"]["experiment"]["record_experience"] = True
    online_confs.append((c, 3))
# runner.p_run_experiments(online_confs)
print("------------------------------------------------")

offline_spec = ["test_cp_off", "test_ab_off"]
offline_confs = []
for name, (on_name, env, v) in zip(offline_spec, online_spec):
    c = make_config(name, env, v)
    c["runner"]["call_"] = FixedBatchRunner.FixedBatchRunner
    c["memory"] = {
        "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer
    }
    offline_confs.append(
        (
            c,
            4,
            os.path.join(
                config.data_dir, f"{env}-{v}", "DQNAgent", on_name, "checkpoints"
            ),
        )
    )
# runner.p_run_experiments(offline_confs)

# cp_off = offline_confs[0][0]
# cp_off["runner"]["call_"] = FixedBatchRunner.FixedBatchRunner
# cp_off["runner"]["experiment"]["redundancy_nr"] = 0
# cp_off["memory"]["_buffers_dir"] = os.path.join(
#     config.data_dir,
#     "-".join(online_spec[0][1:]),
#     "DQNAgent",
#     online_spec[0][0],
#     "checkpoints",
#     "0",
# )
# runner.run_experiment_atomic(cp_off)
