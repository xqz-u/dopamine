from aim import Repo, Run
from thesis import config

# can reload a run if the hash was specified, then pass this instance
# around to continue the run
run = Run(run_hash="egg")
repo = Repo(f"{config.data_dir}/aim_Run")

query = "metric.name in ['loss', 'Train/AverageReturns']"


dfs = {}
for run_metrics_collection in repo.query_metrics(query).iter_runs():
    # print([metric for metric in run_metrics_collection.run.metrics().iter()])
    for metric in run_metrics_collection:
        # Get run params
        dfs[metric.name] = {"params": metric.run[...], "df": metric.dataframe()}
