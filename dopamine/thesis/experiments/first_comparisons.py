import thesis.experiments.dqv_4_runners_cartpole as dqv_cartpole
from thesis.agents import DQNAgent, DQVMaxAgent
from thesis.runner import runner


def gen_confs(envs):
    confs = []
    for env, version in envs:
        for conf_maker in [
            dqv_cartpole.online_conf,
            dqv_cartpole.gb_conf,
            # dqv_cartpole.gb_dqn_exp_conf,
            # dqv_cartpole.fb_dqn_exp_conf,
        ]:
            c = conf_maker()
            c["env"] = {"environment_name": env, "version": version}
            confs.append(c)
    return confs


def dqv_confs():
    return gen_confs([("MountainCar", "v0"), ("Acrobot", "v1")])


def dqvmax_confs():
    confs = gen_confs([("MountainCar", "v0"), ("Acrobot", "v1")])
    for c in confs:
        c["agent"]["call_"] = DQVMaxAgent.DQVMaxAgent
        c["experiment_name"] = c["experiment_name"].replace("dqv", "dqvmax")
    return confs


def dqn_confs():
    confs = gen_confs([("CartPole", "v0"), ("MountainCar", "v0"), ("Acrobot", "v1")])
    for c in confs:
        c["agent"]["call_"] = DQNAgent.DQNAgent
        c["experiment_name"] = c["experiment_name"].replace("dqv", "dqn")
    return confs


def main():
    all_configs = dqn_confs() + dqvmax_confs() + dqv_confs()
    runner.run_multiple_configs(all_configs)


runner.run_multiple_configs(dqn_confs())
# runner.run_multiple_configs(dqvmax_confs())
# runner.run_multiple_configs(dqv_confs())
