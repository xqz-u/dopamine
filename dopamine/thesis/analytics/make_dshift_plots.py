import os
from typing import Tuple

import gym
import pandas as pd
import plotnine as p9
import pymongo
from thesis import constants, utils
from thesis.experiments import (
    dqn_dqvmax_cartpole_acrobot_offline,
    dqv_cartpole_acrobot_offline_vanilla,
)


def dshift_plot(
    df: pd.DataFrame,
    plot_title: str,
    baseline_value: float,
    linecolor="red",
) -> p9.ggplot:
    return (
        p9.ggplot(df, p9.aes("df.index", "Mean"))
        + p9.geom_line(color=linecolor, size=0.7)
        + p9.geom_ribbon(
            p9.aes(ymin="Mean - Std", ymax="Mean + Std"), alpha=0.2, fill=linecolor
        )
        + p9.geom_hline(
            yintercept=baseline_value, color=linecolor, linetype="dashed", size=0.7
        )
        + p9.annotate(
            "label",
            label="$Q^*(s_0, a_0)$",
            x=df.index.max(),
            y=baseline_value - 5,
            ha="right",
            size=15,
        )
        + p9.scale_x_continuous(
            labels=lambda breaks: [f"{int(b/1000)}k" for b in breaks],
            # expand=(0, 0)
        )
        # + p9.scale_y_continuous(
        #     breaks=lambda limits: np.rint(np.linspace(*limits, 10)),
        # )
        + p9.labs(
            title=plot_title, x="Evaluation steps", y="$\\max_{a \\in A}\\ Q(s_0, a)$"
        )
        + p9.theme_bw(base_size=15)
    )


def dshift_df(df: pd.DataFrame) -> pd.DataFrame:
    max_q0_redunds_df = df.pivot_table(
        index=["Global_steps"], columns=["Redundancy"], values=["Max_Q_S0"]
    )
    # NOTE getting this intermediate df since idk how to gather Redundancy
    # back in long format - I can't work with hierarchical indexes yet
    plot_subset_df = df.loc[:, ["Max_Q_S0", "Redundancy", "Global_steps", "QStar_S0"]]
    # and index must be the same for this to work
    plot_subset_df.set_index(["Global_steps"], inplace=True)
    plot_subset_df["Mean"] = max_q0_redunds_df.mean(1)
    plot_subset_df["Std"] = max_q0_redunds_df.std(1)
    return plot_subset_df


def get_experiment_metrics(
    exp_name: str, mongo_client: pymongo.MongoClient
) -> Tuple[pd.DataFrame, str, str]:
    db = mongo_client[exp_name]
    configs_coll = db.configs
    metrics_coll = db.metrics
    exp_config_0 = configs_coll.find_one()
    return (
        pd.DataFrame(list(metrics_coll.find())).drop(columns="_id"),
        exp_config_0["agent"]["call_"],
        exp_config_0["env"],
    )


# TODO facet by agent/env!
def main(mongo_uri="mongodb://localhost:27017/"):
    all_experiments = (
        dqn_dqvmax_cartpole_acrobot_offline.EXPERIMENT_NAMES
        + dqv_cartpole_acrobot_offline_vanilla.EXPERIMENT_NAMES
    )
    client = pymongo.MongoClient(mongo_uri)

    # get all data
    all_exp_metrics = [
        get_experiment_metrics(exp_name, client) for exp_name in all_experiments
    ]
    # subset only evaluation data, where there is the Max_Q_S0 column, and
    # get the baseline QStar value for each env - NOTE could only #envs
    # times... this is hacky but we get the idea
    eval_exp_metrics = [
        [
            df[df["Schedule"] == "eval"],
            f"{agent} on {env}",
            utils.deterministic_discounted_return(gym.make(env)),
            "red" if env == "Acrobot-v1" else "blue",
        ]
        for df, agent, env in all_exp_metrics
    ]
    all_plots = [
        (dshift_plot(dshift_df(df), title, *rest), title)
        for df, title, *rest in eval_exp_metrics
    ]
    for plot, title in all_plots:
        plot.save(filename=os.path.join(constants.pics_dir, title), width=20, height=15)


mongo_uri = constants.xxx_mongo_uri


# if __name__ == "__main__":
# main()
