from dopamine.jax import losses
from thesis.agents import dqv_max
from thesis.runner import reporter, runner

repo = "/home/xqz-u/uni/dopamine/resources/data/test_checkpointing_2"
# cartpole_path = "CartPole-DQVMax"

conf_cartpole = {
    "nets": {
        "qnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e4),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "memory": {},
    "runner": {
        "base_dir": f"{repo}",
        # "schedule": "continuous_train_and_eval",
        # "log_level": logging.INFO,
        "experiment": {
            "seed": 4,
            "steps": 100,
            "iterations": 100,
            "redundancy": 3,
        },
        # "reporters": [
        #     {
        #         "call_": reporter.AimReporter,
        #         "repo": repo,
        #         "experiment": cartpole_path,
        #     }
        # ],
    },
}

run = runner.create_runner(conf_cartpole)
run.run_experiment_with_redundancy()


# from thesis.runner import checkpointing

# ck = checkpointing.Checkpointer(repo)
# agent_data = ck.load_checkpoint(*checkpointing.get_latest_ckpt_number(ck.ckpt_dir))
