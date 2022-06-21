import itertools as it

from thesis import agent, configs, runner
from thesis.experiments import ensemble_all_cartpole_acrobot_offline

agents = [
    agent.DQNEnsemble,
    # agent.DQVMaxEnsemble
]
envs = ["CartPole-v1"]
combs = list(it.product(agents, envs))


online_confs = lambda: [
    c
    | {
        "agent_class": ag,
        "model_maker_fn": configs.dqvmax_model_maker,
        "experiment_name": f"{ag.__name__}_{env}_online_v0",
    }
    for ag, env in combs
    for c in ensemble_all_cartpole_acrobot_offline.online_confs(1)
    if c["env_name"] in envs
]


if __name__ == "__main__":
    on_confs = online_confs()
    runner.run_parallel(on_confs)
