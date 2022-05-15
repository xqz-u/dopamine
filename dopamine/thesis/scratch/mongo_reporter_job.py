from thesis.experiments import pg_time_train_iter_cc
from thesis.reporter import reporter

records = [
    {
        "experiment": "cp_dqvmax_distr_shift",
        "loss": {
            "vfunc_huber_loss": 4500.69482421875,
            "qfunc_huber_loss": 11424.1640625,
        },
        "steps": 1000,
        "q_estimates": 45685.27734375,
        "curr_redundancy": 0,
        "curr_iteration": 0,
        "global_steps": 1000,
        "schedule": "train_and_eval",
        "current_schedule": "train",
    },
    {
        "experiment": "cp_dqvmax_distr_shift",
        "reward": 539,
        "steps": 539,
        "episodes": 4,
        "curr_redundancy": 0,
        "curr_iteration": 0,
        "global_steps": 1000,
        "schedule": "train_and_eval",
        "current_schedule": "eval",
    },
    {
        "experiment": "cp_dqvmax_distr_shift",
        "loss": {
            "vfunc_huber_loss": 2906.219482421875,
            "qfunc_huber_loss": 7997.10009765625,
        },
        "steps": 1000,
        "q_estimates": 45680.56640625,
        "curr_redundancy": 0,
        "curr_iteration": 1,
        "global_steps": 2000,
        "schedule": "train_and_eval",
        "current_schedule": "train",
    },
]

conf = pg_time_train_iter_cc.make_conf("pippo")
conf["runner"]["experiment"]["redundancy_nr"] = 0
mongo_rep = reporter.MongoReporter(
    experiment_name=conf["experiment_name"],
    db_uri="mongodb://s3680622:<my Nestor password>@peregrine.hpc.rug.nl:27017",
    conf=conf,
    buffering=1,
    timeout=5,
    collection_name="test_mongo_job",
)

for r in records:
    mongo_rep(r, None, {})
