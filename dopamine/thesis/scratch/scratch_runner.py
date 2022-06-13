import logging

import gym
import optax
from dopamine.discrete_domains import gym_lib
from dopamine.jax import losses
from jax import numpy as jnp
from thesis import (
    agent,
    constants,
    custom_pytrees,
    exploration,
    instantiators,
    memory,
    networks,
    reporter,
    runner,
    utils,
)
from thesis.agent import utils as agent_utils

utils.setup_root_logging(logging.DEBUG)


env = gym.make("CartPole-v1")
env_preproc = constants.env_preproc_info("CartPole", "v1")["preproc"]
env_info = constants.env_info(env)
env_state_shape = env_info["observation_shape"] + (1,)
env_state_example = jnp.ones(env_state_shape)

offline_mem = memory.load_offline_buffers(
    f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
    [0],
    load_parallel=False,
    **env_info,
    **constants.default_memory_args,
)
mem = memory.OutOfGraphReplayBuffer(**constants.default_memory_args, **env_info)

experience_batch = agent_utils.sample_replay_buffer(offline_mem, batch_size=8)

rng = custom_pytrees.PRNGKeyWrap(42)

q_mlp = (networks.MLP, {"features": 2, "hiddens": (4,), **env_preproc})

v_mlp = (networks.MLP, {"features": 1, "hiddens": (4,), **env_preproc})
v_ensemble = (networks.EnsembledNet, {"model": v_mlp, "n_heads": 2})

adam = lambda: optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4})

agent_common_args = lambda: {
    **env_info,
    "rng": rng,
    "sync_weights_every": 1,
    "min_replay_history": 100,
    "policy_evaluator": exploration.Egreedy(rng=rng, num_actions=env.action_space.n),
}

dqn_ag = agent.DQN(
    Q_model_def=instantiators.create_model_TS_def(q_mlp, adam(), losses.mse_loss),
    **agent_common_args(),
    memory=mem,
)

# TODO hustle to give memory here, when in reality it is determined by
# the type of runner!
dqv_ensemble_ag = agent.DQVEnsemble(
    V_model_def=instantiators.create_model_TS_def(v_ensemble, adam(), losses.mse_loss),
    Q_model_def=instantiators.create_model_TS_def(q_mlp, adam(), losses.mse_loss),
    **agent_common_args(),
    memory=mem,
    # memory=offline_mem,
)

# run_offline = runner.FixedBatchRunner(
#     experiment_name="pippo",
#     agent=dqv_ensemble_ag,
#     iterations=3,
#     steps=500,
#     eval_steps=100,
#     eval_period=1,
#     env=gym_lib.GymPreprocessing(env),
#     on_policy_eval=[agent_utils.t0_max_q_callback],
#     checkpoint_dir=os.path.join(constants.scratch_data_dir, "scratch_runner"),
# )
# run_offline.run()


mongo_rep = reporter.MongoReporter(
    experiment_name="pippo_online",
    **{
        # default db_uri,
        "buffering": 4,
        "db_name": "thesis_test",
        "collection_name": "pippo_online",
    },
)

aim_rep = reporter.AimReporter(
    repo=str(constants.scratch_data_dir), experiment_name="pippo_online"
)

run_online = runner.OnlineRunner(
    # schedule="train",
    experiment_name="pippo_online",
    # reporters=[mongo_rep, aim_rep],
    # agent=dqn_ag,
    agent=dqv_ensemble_ag,
    iterations=10,
    steps=500,
    eval_steps=100,
    eval_period=1,
    env=gym_lib.GymPreprocessing(env),
    on_policy_eval=[agent_utils.t0_max_q_callback],
    checkpoint_base_dir=constants.scratch_data_dir,
    # record_experience=True,
)
run_online.run()
