import logging
import operator

import gym
import jax
import optax
from dopamine.jax import losses
from jax import numpy as jnp
from thesis import constants, custom_pytrees, networks, offline_memory, utils
from thesis.agents import dqn, dqv, dqv_max
from thesis.agents import utils as agent_utils
from thesis.exploration import egreedy

utils.setup_root_logging(logging.DEBUG)

env = gym.make("CartPole-v1")
env_preproc = constants.env_preproc_info("CartPole", "v1")["preproc"]
env_info = constants.env_info(env)
env_state_shape = env_info["observation_shape"] + (1,)
env_state_example = jnp.ones(env_state_shape)


mem = offline_memory.OfflineOutOfGraphReplayBuffer(
    f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
    [0],
    load_parallel=False,
    **env_info,
    **constants.default_memory_args,
)

experience_batch = agent_utils.sample_replay_buffer(mem, batch_size=8)


exp_rng = custom_pytrees.PRNGKeyWrap(42)

dqn_mlp = (networks.MLP, {"features": 2, "hiddens": (4,)})
dqn_ensemble = (networks.EnsembledNet, {"model": dqn_mlp, "n_heads": 3})

adam = optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4})

dqn_ts_spec = (dqn_mlp, adam, losses.mse_loss)
dqn_ensemble_ts_spec = (dqn_ensemble, adam, losses.mse_loss)


dqn_common_args = lambda: {
    "memory": mem,
    **env_info,
    "rng": exp_rng,
    "sync_weights_every": 1,
    "policy_evaluator": egreedy.Egreedy(rng=exp_rng, num_actions=env.action_space.n),
}


ag = dqn.DQN(
    Q_model_def=(agent_utils.build_models(dqn_ts_spec[0]),) + dqn_ts_spec[1:],
    **dqn_common_args(),
)
ag_ensemble = dqn.DQNEnsemble(
    Q_model_def=(agent_utils.build_models(dqn_ensemble_ts_spec[0]),)
    + dqn_ensemble_ts_spec[1:],
    **dqn_common_args(),
)


v_mlp = (networks.MLP, {"features": 1, "hiddens": (4,)})
v_ensemble = (networks.EnsembledNet, {"model": v_mlp, "n_heads": 3})

ag_dqv = dqv.DQV(
    V_model_def=(agent_utils.build_models(v_mlp),) + dqn_ts_spec[1:],
    Q_model_def=(agent_utils.build_models(dqn_ts_spec[0]),) + dqn_ts_spec[1:],
    **dqn_common_args(),
)
ag_dqv_ensemble = dqv.DQVEnsemble(
    V_model_def=(agent_utils.build_models(v_ensemble),) + dqn_ts_spec[1:],
    Q_model_def=(agent_utils.build_models(dqn_ts_spec[0]),) + dqn_ts_spec[1:],
    **dqn_common_args(),
)

ag_dqvmax = dqv_max.DQVMax(
    V_model_def=(agent_utils.build_models(v_mlp),) + dqn_ts_spec[1:],
    Q_model_def=(agent_utils.build_models(dqn_ts_spec[0]),) + dqn_ts_spec[1:],
    **dqn_common_args(),
)
ag_dqvmax_ensemble = dqv_max.DQVMaxEnsemble(
    V_model_def=(agent_utils.build_models(v_ensemble),) + dqn_ts_spec[1:],
    Q_model_def=(agent_utils.build_models(dqn_ensemble_ts_spec[0]),)
    + dqn_ensemble_ts_spec[1:],
    **dqn_common_args(),
)


for agent in [ag, ag_ensemble, ag_dqv, ag_dqv_ensemble, ag_dqvmax, ag_dqvmax_ensemble]:
    print(f"{agent.__class__}: {agent.learn()}")
    print("_S__________________________________________________")


for agent in [ag, ag_ensemble, ag_dqv, ag_dqv_ensemble, ag_dqvmax, ag_dqvmax_ensemble]:
    run_steps, n_episodes, run_rwd = 0, 0, 0.0
    obs, done = env.reset(), False
    train_dict = {"reward": 0.0, "steps": 0, **agent.initial_train_dict}
    while not done:
        env.render()
        action, (max_q,) = agent.select_action(obs, "train")
        obs, reward, done, _ = env.step(action)
        agent.record_trajectory(reward, done)
        train_dict = jax.tree_map(
            operator.add, train_dict, {"reward": reward, "steps": 1, **agent.learn()}
        )
        print(train_dict)

        print(max_q)
    print(f"DONE {agent.__class__} _________________________________________________")
env.close()
