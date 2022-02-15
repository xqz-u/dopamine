import gin
from dopamine.discrete_domains import run_experiment
from thesis import config, utils


# NOTE should call utils.data_dir_from_conf (unique_data_dir does not
# exist anymore), but I have the config as a gin file
def main():
    dqn_cartpole_config_file = (
        f"{config.dopamine_dir}/jax/agents/dqn/configs/dqn_cartpole.gin"
    )
    gin.parse_config_file(dqn_cartpole_config_file)
    logdir = utils.unique_data_dir("dqn_cartpole_online_train")
    runner = run_experiment.create_runner(base_dir=logdir)
    runner.run_experiment()
