import optax
from dopamine.jax import losses
from thesis import config
from thesis.agents import agents
from thesis.reporter import reporter

model_conf = {
    "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss": losses.huber_loss,
}
make_config = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qnet": model_conf, "vnet": model_conf},
    "exploration": {},
    "agent": {
        "call_": agents.DQVMaxAgent.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e3),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "memory": {"replay_capacity": int(5e4)},
    "runner": {
        # "call_": runner.GrowingBatchRunner,
        "schedule": "train",
        "experiment": {
            "seed": 4,
            "steps": 600,
            "iterations": 10,
            "redundancy": 1,
        },
        "reporters": {
            "mongo": {
                "call_": reporter.MongoReporter,
                "db_name": "test_database",
                "collection_name": "test_collection",
                "buffering": 10,
            },
            "aim": {"call_": reporter.AimReporter, "repo": str(config.aim_dir)},
        },
    },
}


def main():
    from thesis import utils
    from thesis.runner import runner

    conf = make_config("test_mongo")
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    manager = runner.create_runner(conf)
    manager.run_experiment_with_redundancy()


# main()
