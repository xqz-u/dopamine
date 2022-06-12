import thesis.experiments.dqv_4_runners_cartpole as dqv_cartpole
from thesis.agents.DQNAgent import DQNAgent
from thesis.runner import runner

# def test():
#     from thesis import config, utils

#     conf = online_conf()
#     conf["runner"].pop("log_level", None)
#     conf["runner"]["reporters"][0]["repo"] = str(config.test_dir)
#     utils.data_dir_from_conf(conf["experiment_name"], conf, basedir=config.test_dir)
#     run = runner.create_runner(conf)
#     run.run_experiment()


# test()


def main():
    all_configs = [
        conf_maker()
        for conf_maker in [
            dqv_cartpole.online_conf,
            dqv_cartpole.gb_conf,
            dqv_cartpole.gb_dqn_exp_conf,
            dqv_cartpole.fb_dqn_exp_conf,
        ]
    ]
    for conf in all_configs:
        conf["agent"]["call_"] = DQNAgent
        conf["experiment_name"] = conf["experiment_name"].replace("dqv", "dqn")
    runner.run_multiple_configs(all_configs)


# main()
