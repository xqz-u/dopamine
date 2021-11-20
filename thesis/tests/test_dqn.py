import os
from pprint import pp

import gin
import matplotlib.pyplot as plt

from dopamine.colab import utils as colab_u
from dopamine.discrete_domains import run_experiment

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")

cartpole_conf = os.path.join(THESIS, "dqn_cartpole.gin")
gin.parse_config_file(cartpole_conf)

runner = run_experiment.create_runner(THESIS)
runner.run_experiment()

log_dir = os.path.join(THESIS, "logs")
redun = 0  # will load latest log file if not specified
# stats, redun_n = colab_u.load_statistics(log_ir, redun)
stats, redun_n = colab_u.load_statistics(log_dir)
# assert redun == redun_n
pp(stats)

# the parameters in the .gin file already write a summary statistics for some of
# the logged stats; in any case, they could be obtained with a call like the
# one below
summ = colab_u.summarize_data(stats, ["train_episode_returns"])

# plt.plot(stats["iteration_0"]["train_episode_returns"], label="episode returns")
# plt.plot()
# plt.title("DQN on CartPole-v0, 1 iteration")
# plt.xlabel("Episode")
# plt.ylabel("Return")
# plt.legend()
# plt.show()


plt.plot(summ["train_episode_returns"], label="episode returns")
plt.plot()
plt.title(f"DQN on CartPole-v0, {redun_n + 1} iterations")
plt.xlabel("Iteration")
plt.ylabel("Return")
plt.legend()
plt.show()


for iter_, d in stats.items():
    print(f"{iter_} episode lengths: {d['train_episode_lengths']}")
