import pandas as pd
import pymongo
from scipy import stats
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
        test = (
            stats.mannwhitneyu
            if not (check_normality(ag_data[m]) and check_normality(ens_ag_data[m]))
            else stats.ttest_ind
        )
        test_res = test(ag_data[m], ens_ag_data[m])
        print(f"**{test.__name__}**: {test_res}")
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
grouped_abl_data = abl_eval_data.groupby("Agent")
print("\n\nEnsemble DQV-Max ablations diff:")
for m in ["Reward", "Max_Q_S0"]:
    seqs = grouped_abl_data[m].agg(list)
    test = stats.f_oneway
    for s in seqs:
        if not check_normality(s):
            test = stats.kruskal
    print(f"\t{m.upper()} p {test.__name__}: {test(*seqs)}")


# |                  | CartPole-v1 |          |          |         | Acrobot-v1 |         |          |       |
# |------------------+-------------+----------+----------+---------+------------+---------+----------+-------|
# |                  |      Reward |          |  Q-Value |         | Reward     |         | Q-Value  |       |
# |                  |        Mean |      Var |     Mean |     Var | Mean       |     Var | Mean     |   Var |
# |------------------+-------------+----------+----------+---------+------------+---------+----------+-------|
# | DQN              |      440.36 | 15540.87 | 208.71 * | 8755.07 | -70.48 *   | 2533.65 | -45.22 * | 14.57 |
# | Ensemble-DQN     |      410.55 | 23157.11 |   163.70 | 2330.90 | -76.02     | 2949.21 | -45.78   | 15.70 |
# | DQV              |      489.09 |  2473.61 |    79.68 |   79.68 | -70.86 *   | 2032.89 | -51.45 * | 19.04 |
# | Ensemble-DQV     |      475.88 |  6657.30 |    79.56 |   81.74 | -82.43     | 5457.53 | -51.60   | 31.53 |
# | DQV-Max          |      445.77 | 14386.91 |    99.70 |  109.31 | -69.90 *   | 1718.89 | -48.06 * | 10.45 |
# | Ensemble-DQV-Max |      430.85 | 18011.71 |    98.12 |   98.26 | -77.73     | 3544.01 | -48.42   | 11.61 |


# Ensemble DQV-Max ablations diff:
#       ****sample not normal! p: 0.0
#       ****sample not normal! p: 0.0
#       ****sample not normal! p: 0.0
#       REWARD p kruskal: KruskalResult(statistic=1.364962138148842, pvalue=0.5053615986165446)
#       ****sample not normal! p: 0.0
#       ****sample not normal! p: 0.0
#       ****sample not normal! p: 0.0
#       MAX_Q_S0 p kruskal: KruskalResult(statistic=0.30792649762406654, pvalue=0.8573035273555845)
