import os
from typing import List

import gym
import pandas as pd
import pymongo
import seaborn as sns
from matplotlib import pyplot as plt
from thesis import constants, utils
from thesis.experiments import all_cartpole_acrobot_offline

# NOTE should be gathered programmatically
envs = ["CartPole-v1", "Acrobot-v1"]
gym_envs = [gym.make(env) for env in envs]
BASELINES = {
    env_name: {
        "Max_Q_S0": utils.deterministic_discounted_return(env),
        "Reward": opt_reward,
        # "Max_Q_S0_ewm": utils.deterministic_discounted_return(env),
        # "Reward_ewm": opt_reward,
    }
    for env_name, env, opt_reward in zip(envs, gym_envs, (500, -100))
}


def get_experiment_metrics(
    exp_name: str, mongo_client: pymongo.MongoClient
) -> pd.DataFrame:
    db = mongo_client[exp_name]
    return pd.DataFrame(list(db.metrics.find())).drop(columns="_id")


def get_data(exp_names: List[str], mongo_client: pymongo.MongoClient) -> pd.DataFrame:
    return pd.concat(
        [get_experiment_metrics(exp_name, mongo_client) for exp_name in exp_names]
    )


def plot_learners_per_env(
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    ymetric: str,
    xlabel: str = None,
    ylabel: str = None,
) -> plt.Axes:
    env_name = data["Env"].unique()[0]
    linewidth = 7
    sns.lineplot(
        **{
            "data": data.reset_index(),
            "ax": ax,
            "x": "Global_steps",
            "y": ymetric,
            "hue": "Agent",
            "ci": "sd",
            "linewidth": linewidth,
        }
    )
    ax.axhline(
        y=BASELINES[env_name][ymetric],
        linestyle="--",
        color="black",
        linewidth=linewidth,
        label="Baseline",
    )
    # indicate * 10e3 suffix - not on datapoint 0
    labs = [
        f"{(int(float(t / 1000)))}{'e3' if t else ''}" for t in ax.get_xticks().tolist()
    ]
    ax.set_xticklabels(labs)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(env_name)
    ax.get_legend().remove()
    return ax


# mongo_uri = "mongodb://localhost:27017/"
mongo_uri = constants.xxx_mongo_uri
exp_suffix = "_pres_"

client = pymongo.MongoClient(mongo_uri)

# exp_names = all_cartpole_acrobot_offline.EXPERIMENT_NAMES()
# exp_names = [
#     name
#     for name in client.list_database_names()
#     if exp_suffix in name and "fake" not in name
# ]
exp_names = [
    f"MultiHeadEnsemble{algo}_{env}_pres"
    for algo in ["DQN", "DQVMax", "DQVTiny"]
    for env in ["CartPole-v1", "Acrobot-v1"]
]

data = get_data(exp_names, client)

eval_data = data[data["Schedule"] == "eval"]

# rename the ensembles for better legend
eval_data["Agent"] = eval_data["Agent"].apply(
    lambda s: s.replace("Ensemble", "").replace("Tiny", "")
)

# global matplotilb parameters, should better be set per plot...
plt.rcParams["font.size"] = 50
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

fig, axes = plt.subplots(2, 2, figsize=(60, 40))

cp_df = eval_data[eval_data["Env"] == "CartPole-v1"].copy()
ab_df = eval_data[eval_data["Env"] == "Acrobot-v1"].copy()

cp_df.loc[:, "Reward_ewm"] = cp_df["Reward"].ewm(com=0.7).mean()
ab_df.loc[:, "Reward_ewm"] = ab_df["Reward"].ewm(com=0.7).mean()
cp_df.loc[:, "Max_Q_S0_ewm"] = cp_df["Max_Q_S0"].ewm(com=0.7).mean()
ab_df.loc[:, "Max_Q_S0_ewm"] = ab_df["Max_Q_S0"].ewm(com=0.7).mean()

plots = [
    plot_learners_per_env(
        fig,
        axes[j][i],
        df,
        ymetric,
        xlabel="Evaluation steps",
        ylabel=ylabel,
    )
    for i, df in enumerate([cp_df, ab_df])
    for j, (ymetric, ylabel) in enumerate(
        [("Max_Q_S0", "Value Estimates"), ("Reward", "Reward")]
        # [("Max_Q_S0_ewm", "Value Estimates"), ("Reward_ewm", "Reward")]
    )
]

# put one single legend for both plots
labels, handles = axes[0][0].get_legend_handles_labels()
leg = fig.legend(labels, handles, loc="upper center", ncol=len(labels))
for line in leg.get_lines():
    line.set_linewidth(8)

plt.tight_layout()

plt.subplots_adjust(top=0.93)

plt.savefig(
    os.path.join(
        constants.resources_dir,
        "symposium",
        # "dshift_plots_normal.png"
        # "dshift_plots_normal_ewm_07.png"
        # "dshift_plots_ensembles_ewm_07.png"
        "dshift_plots_ensembles.png",
    ),
    transparent=True,
)
