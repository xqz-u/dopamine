import functools as ft
import os

import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import seaborn as sns
from scipy import stats
from thesis import constants
from thesis.analytics import plot_scratch

alpha = 0.01

exp_name_stitcher = lambda ags, suff, pre="": [
    f"{pre}{a}_{e}{suff}" for a in ags for e in plot_scratch.envs
]

exp_names = exp_name_stitcher(
    ["DQN", "DQVMax", "DQVTiny"], "_pres", pre="MultiHeadEnsemble"
) + exp_name_stitcher(["DQN", "DQVMax", "DQV"], "_offline_v1")

all_agents = [e.split("_")[0] for e in exp_names[::2]]
half = len(all_agents) // 2
compare_agents = list(zip(all_agents[:half], all_agents[half:]))

client = pymongo.MongoClient(plot_scratch.mongo_uri)

data = plot_scratch.get_data(exp_names, client)
eval_data = data[data["Schedule"] == "eval"]


def check_normality(s):
    if (p := stats.normaltest(s).pvalue) <= alpha:
        print(f"\t****sample not normal! p: {p}")
        return False
    return True


# from wikipedia:
# Otherwise, if both the dispersions and shapes of the distribution of
# both samples differ, the Mann-Whitney U test fails a test of
# medians. It is possible to show examples where medians are numerically
# equal while the test rejects the null hypothesis with a small
# p-value.[4] [5]
# also, see https://arxiv.org/abs/1904.06979
# in essence, use Welch t-test (ttest with relaxed assumption of equal
# pop. variance) which seems more robust to false positives; although
# my distributions are not normal, it should still work well enough
def ttest_diffs(data: pd.DataFrame, ag: str, ens_ag: str, env: str):
    env_data = data[data["Env"] == env]
    ag_data = env_data[env_data["Agent"] == ag]
    ens_ag_data = env_data[env_data["Agent"] == ens_ag]
    print(f"Agents: {ag} - {ens_ag}\nEnv: {env}")
    for m in ["Reward", "Max_Q_S0"]:
        print()
        ag_summ, ens_ag_summ = stats.describe(ag_data[m]), stats.describe(
            ens_ag_data[m]
        )
        print(
            f"{m.upper()}\nmeans: {ag_summ.mean} | {ens_ag_summ.mean}\nvars: {ag_summ.variance} | {ens_ag_summ.variance}"
        )
        normality = check_normality(ag_data[m]) and check_normality(ens_ag_data[m])
        pargs = {"alternative": "greater"}
        # if not normality:
        if False:
            test = stats.mannwhitneyu
        else:
            test = stats.ttest_ind
            pargs.update({"equal_var": False})
            # test = stats.mannwhitneyu
            # test = stats.ks_2samp
        test_name = test.__name__
        test_res = ft.partial(test, **pargs)(ag_data[m], ens_ag_data[m])
        print(f"**{test_name}**: {test_res}")
        if test_res.pvalue <= alpha:
            print("___SIGNIFICANT DIFF___")


# compare each offline ensemble agent against its standard version
for ens_ag, ag in compare_agents:
    print("---------------------------")
    for e in plot_scratch.envs:
        print(">>>>>")
        ttest_diffs(eval_data, ag, ens_ag, e)

# compare ensemble dqvmax against its ablated versions
dqvmax_ablation_exps = exp_name_stitcher(
    ["DQVMax", "DQVMaxOnQ", "DQVMaxOnV"], "_pres", pre="MultiHeadEnsemble"
)
abl_data = plot_scratch.get_data(dqvmax_ablation_exps, client)
abl_eval_data = abl_data[abl_data["Schedule"] == "eval"]
print("\n\nEnsemble DQV-Max ablations diff:")
for env in ["CartPole-v1", "Acrobot-v1"]:
    ag_env_data = abl_eval_data[abl_eval_data.Env == env].groupby("Agent")
    for m in ["Reward", "Max_Q_S0"]:
        seqs = ag_env_data[m].agg(list)
        test = stats.f_oneway
        for s in seqs:
            if not check_normality(s):
                test = stats.kruskal
        print(f"\t{env} {m.upper()} p {test.__name__}: {test(*seqs)}")
    print()


def add_dist_plot(all_data, agents_pair, env_axes, metric: str, lgd_loc):
    agents_data = all_data[all_data.Agent.isin(agents_pair)].reset_index()
    for ax, env, loc in zip(env_axes, ["CartPole-v1", "Acrobot-v1"], lgd_loc):
        sns.kdeplot(
            data=agents_data,
            x=agents_data[agents_data.Env == env][metric],
            hue="Agent",
            fill=True,
            ax=ax,
        )
        ax.set_title(env)
        sns.move_legend(ax, loc)


def make_lgd_pos(metric, first, n):
    if metric == "Max_Q_S0":
        return (
            [["best", "best"]] + [["upper left", "best"]] * 2
            if first
            else [["upper left", "best"]] * n
        )
    if metric == "Reward":
        return [["upper left", "upper left"]] * n


if __name__ == "__main__":
    plt.rcParams["font.size"] = 55
    plt.rcParams["axes.linewidth"] = 3
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    rplc = lambda s: s.replace("MultiHead", "").replace("Tiny", "")

    eval_data["Agent"] = eval_data["Agent"].apply(rplc)
    mod_compare_agents = [(rplc(e), std) for e, std in compare_agents]

    # plot the distribution of Rewards and Max_Q_S0 of each ensemble
    # agent against its base version, across environments
    for metric, title in zip(["Max_Q_S0", "Reward"], ["all_qv_dist", "all_rwd_dist"]):
        fig, axes = plt.subplots(2, 3, figsize=(60, 40))
        for i, (ag_pair, lgd_pos) in enumerate(
            zip(mod_compare_agents, make_lgd_pos(metric, True, len(mod_compare_agents)))
        ):
            add_dist_plot(eval_data, ag_pair, axes[:, i], metric, lgd_pos)
            plt.tight_layout()
            plt.savefig(
                os.path.join(constants.resources_dir, "paper/img", f"{title}.png"),
                transparent=True,
            )

    # same as above, but for ablations of EnsembleDQVMax against
    # EnsembleDQVMax
    abl_eval_data["Agent"] = abl_eval_data["Agent"].apply(rplc)
    for metric, title in zip(
        ["Max_Q_S0", "Reward"], ["dqvmax_abl_qv_dist", "dqvmax_abl_rwd_dist"]
    ):
        fig, axes = plt.subplots(2, 2, figsize=(60, 40))
        lgd_poss = make_lgd_pos(metric, False, 2)
        add_dist_plot(
            abl_eval_data,
            ["EnsembleDQVMax", "EnsembleDQVMaxOnQ"],
            axes[:, 0],
            metric,
            lgd_poss[0],
        )
        add_dist_plot(
            abl_eval_data,
            ["EnsembleDQVMax", "EnsembleDQVMaxOnV"],
            axes[:, 1],
            metric,
            lgd_poss[1],
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(constants.resources_dir, "paper/img", f"{title}.png"),
            transparent=True,
        )
