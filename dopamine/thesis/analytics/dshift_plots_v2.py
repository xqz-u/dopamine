import os
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import seaborn as sns
from thesis import constants, utils
from thesis.experiments import all_cartpole_acrobot_offline

# darkgrid by default
# sns.set_theme()


def plot_add_qstar_s0(data, **kws):
    env_name = data["Env"].unique()[0]
    ax = plt.gca()
    ax.axhline(
        y=utils.deterministic_discounted_return(gym.make(env_name)),
        ls="--",
        color="grey",
    )


def plot_tabulated(df: pd.DataFrame, **args) -> Tuple[sns.FacetGrid]:
    maxq0_ax = sns.relplot(**args, y="Max_Q_S0")
    # add baseline q_s0 for env
    maxq0_ax.map_dataframe(plot_add_qstar_s0)
    maxq0_ax.tight_layout()
    reward_ax = sns.relplot(**args, y="Reward", legend=False)
    reward_ax.tight_layout()
    return maxq0_ax, reward_ax


# TODO plot all the different runs instead of aggregations for a metric,
# and make the mean one bold; check optimistic_offline_rl for reference
# TODO label for baseline q0
# TODO space between legend and titles
# TODO better titles and x values
# TODO font size and face
# TODO environment name as proper title
# NOTE I can't stack the two facet plots vertically with matplotlib
# currently
def plot_by_env(df: pd.DataFrame, **args) -> Tuple[sns.FacetGrid]:
    maxq0_ax = sns.relplot(**args, y="Max_Q_S0")
    # add baseline q_s0 for env
    maxq0_ax.map_dataframe(plot_add_qstar_s0)
    sns.move_legend(
        maxq0_ax,
        "upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )
    maxq0_ax.tight_layout()

    reward_ax = sns.relplot(**args, y="Reward", legend=False)
    reward_ax.tight_layout()
    return maxq0_ax, reward_ax


# NOTE if experiments were saved under the same collection, I could do
# a collection.find({"experiment_name": {"$in": [...]}})
def get_experiment_metrics(
    exp_name: str, mongo_client: pymongo.MongoClient
) -> pd.DataFrame:
    db = mongo_client[exp_name]
    return pd.DataFrame(list(db.metrics.find())).drop(columns="_id")


def get_data(exp_names: List[str], mongo_uri: str) -> list:
    client = pymongo.MongoClient(mongo_uri)
    return pd.concat(
        [get_experiment_metrics(exp_name, client) for exp_name in exp_names]
    )


def save_plot(imgs: List[sns.FacetGrid], titles: List[str]):
    for img, title in zip(imgs, titles):
        img.figure.savefig(os.path.join(constants.pics_dir, title))


def main(
    exp_names: List[str], mongo_uri: str = "mongodb://localhost:27017/", save=False
):
    metrics = get_data(exp_names, mongo_uri)
    # subset only evaluation data, where there is the Max_Q_S0 column
    eval_metrics = metrics[metrics["Schedule"] == "eval"]

    args = {
        "data": eval_metrics.reset_index(),
        "x": "Global_steps",
        "hue": "Agent",
        "col": "Env",
        "ci": "sd",
        "kind": "line",
        "facet_kws": dict(legend_out=False),
        "height": 8,
        "aspect": 0.9,
    }
    p_qmax, p_reward = plot_by_env(eval_metrics, **args)
    if save:
        save_plot(
            [p_qmax, p_reward], ["qmax_s0_offline.png", "reward_eval_offline.png"]
        )

    args = args | {
        "hue": "Env",
        "row": "Env",
        "col": "Agent",
    }
    p_qmax_facets, p_reward_facets = plot_tabulated(eval_metrics, **args)
    if save:
        save_plot(
            [p_qmax_facets, p_reward_facets],
            ["qmax_s0_offline_facets.png", "reward_eval_offline_facets.png"],
        )
    return p_qmax, p_qmax_facets, p_reward, p_qmax_facets


if __name__ == "__main__":
    # main(all_cartpole_acrobot_offline.EXPERIMENT_NAMES())
    main(
        all_cartpole_acrobot_offline.EXPERIMENT_NAMES(),
        mongo_uri=constants.xxx_mongo_uri,
        save=True,
    )
