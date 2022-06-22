import os
from typing import List

import gym
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import seaborn as sns
from thesis import constants, utils
from thesis.experiments import all_cartpole_acrobot_offline


# NOTE if experiments were saved under the same collection, I could do
# a collection.find({"experiment_name": {"$in": [...]}})
def get_experiment_metrics(
    exp_name: str, mongo_client: pymongo.MongoClient
) -> pd.DataFrame:
    db = mongo_client[exp_name]
    return pd.DataFrame(list(db.metrics.find())).drop(columns="_id")


def main(exp_names: List[str], mongo_uri="mongodb://localhost:27017/") -> list:
    client = pymongo.MongoClient(mongo_uri)
    return pd.concat(
        [get_experiment_metrics(exp_name, client) for exp_name in exp_names]
    )


def plot_add_qstar_s0(data, **kws):
    env_name = data["Env"].unique()[0]
    ax = plt.gca()
    ax.axhline(
        y=utils.deterministic_discounted_return(gym.make(env_name)),
        ls="--",
        color="grey",
    )


experiments = all_cartpole_acrobot_offline.EXPERIMENT_NAMES()
metrics = main(experiments)
# subset only evaluation data, where there is the Max_Q_S0 column
eval_metrics = metrics[metrics["Schedule"] == "eval"]


# NOTE when in long format, seaborn already computes the mean of
# observations at the x variable; this happens when indexing by
# global_steps in my case
# TODO annotate hline, or put in legend
sns.set_theme(style="darkgrid")
ax = sns.relplot(
    data=eval_metrics.reset_index(),
    x="Global_steps",
    y="Max_Q_S0",
    kind="line",
    hue="Agent",
    col="Env",
    ci="sd",
)
ax.map_dataframe(plot_add_qstar_s0)
# label every facet axis
# for axis in p.axes.flat:
#     axis.tick_params(labelleft=True)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
)
ax.tight_layout()

plt.savefig(os.path.join(constants.pics_dir, "test_pic.png"))


# cp_dqn = eval_metrics[
#     (eval_metrics["Agent"] == "DQN") & (eval_metrics["Env"] == "CartPole-v1")
# ]

# cp_dqn_plot = sns.relplot(
#     data=cp_dqn.reset_index(),
#     x="Global_steps",
#     y="Max_Q_S0",
#     kind="line",
#     ci="sd",
# )
# cp_dqn_plot.map_dataframe(plot_add_qstar_s0)
# cp_dqn_plot.tight_layout()
# cp_dqn_plot.fig.show()
