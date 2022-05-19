import os

from thesis import config, constants
from thesis.agents import agents
from thesis.runner import Runner, runner

make_conf = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": config.classic_control_mlp_huberloss_adam,
        "vfunc": config.classic_control_mlp_huberloss_adam,
    },
    "exploration": config.egreedy_exploration,
    "agent": config.make_batch_rl_agent(agents.DQVMaxAgent),
    "memory": config.make_batch_rl_memory(),
    "env": config.make_env("CartPole", "v1"),
    "reporters": config.make_reporters(exp_name),
    "runner": {
        "call_": runner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": int(1e3),
            "iterations": 10,
            "eval_period": 1,
        },
    },
}


conf = make_conf("scratchpad")
conf, *_ = runner.expand_conf(
    conf,
    1,
    os.path.join(
        constants.data_dir,
        "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
    ),
)
conf["reporters"]["mongo"]["buffering"] = 1

run = runner.build_runner(conf, constants.scratch_data_dir)
run.run_experiment()
# ag = run.agent

# loss = ag.learn()
# loss["steps"] = ag.training_steps
# loss_agg_metrics = Runner.aggregate_losses(ag.loss_names, loss)

# ep_eval_metrics = run.eval_one_episode()

# iter_eval_metrics = run.eval_iteration()
