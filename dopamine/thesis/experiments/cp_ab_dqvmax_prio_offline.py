import os

from thesis import config
from thesis.memory import prio_offline_memory
from thesis.runner import runner
from thesis.scratch import test_off_mem


def main():
    conf_cp = test_off_mem.conf_cartpole_dqvmax_offline("cp_dqvmax_off_prio")
    conf_ab = test_off_mem.conf_cartpole_dqvmax_offline("ab_dqvmax_off_prio")
    conf_ab["env"] = ({"environment_name": "Acrobot", "version": "v1"},)

    dqn_offline_data = [
        os.path.join(
            config.data_dir,
            env,
            f"DQNAgent/{abbr}_dqn_full_experience_%%/checkpoints/full_experience",
        )
        for env, abbr in [("CartPole-v1", "cp"), ("Acrobot-v1", "ab")]
    ]

    for c, off_data in zip([conf_cp, conf_ab], dqn_offline_data):
        c["memory"].update(
            {
                "call_": prio_offline_memory.PrioritizedOfflineOutOfGraphReplayBuffer,
                "_buffers_dir": off_data,
            }
        )
        c["reporters"]["aim"]["repo"] = str(config.data_dir)

    redundancy = 3
    confs = [
        runner.add_offline_buffers(runner.add_redundancies(c, redundancy), data)
        for c, data in zip([conf_cp, conf_ab], dqn_offline_data)
    ]
    runner.p_run_experiments([c for exp_conf in confs for c in exp_conf])


# main()
