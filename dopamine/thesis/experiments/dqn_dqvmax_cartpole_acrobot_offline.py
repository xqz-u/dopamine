import itertools as it

from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner

exp_name_fn = (
    lambda ag, env, prefix="": f"{prefix}{utils.callable_name_getter(ag)}_{env}_offline"
)


EXPERIMENT_NAMES = [
    exp_name_fn(*arg)
    for arg in it.product([agent.DQN, agent.DQVMax], ["CartPole-v1", "Acrobot-v1"])
]

confs = [
    c
    for agent_class, model_maker in [
        [agent.DQN, configs.dqn_model_maker],
        [agent.DQVMax, configs.dqvmax_model_maker],
    ]
    for env_name, offline_buff_dir in [
        ["CartPole-v1", experiments.dqn_cartpole_replay_buffers_root],
        ["Acrobot-v1", experiments.dqn_acrobot_replay_buffers_root],
    ]
    for c in [
        {
            "seed": experiments.DEFAULT_SEED + i,
            "redundancy": i,
            "agent_class": agent_class,
            "env_name": env_name,
            "offline_root_data_dir": offline_buff_dir,
            "experiment_name": exp_name_fn(agent_class, env_name),
            "model_maker_fn": model_maker,
            # "logs_base_dir": constants.data_dir,
            "logs_base_dir": constants.scratch_data_dir,
            "experiment": {
                "iterations": 10,
                "steps": 200,
                "eval_steps": 200,
                "eval_period": 2,
            },
        }
        for i in range(experiments.DEFAULT_REDUNDANCY)
    ]
]


if __name__ == "__main__":
    runner.run_parallel(confs, runner.FixedBatchRunner)
