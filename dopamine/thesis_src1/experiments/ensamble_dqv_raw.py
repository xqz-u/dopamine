import functools as ft
from typing import List

import gym
import jax
import numpy as np
import optax
from dopamine.jax import losses
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import constants, custom_pytrees, networks
from thesis.agents import agent_utils
from thesis.memory import offline_memory

SEED = 5
ENSEMBLE_HEADS = 2
MLP_HIDDEN_SIZE = (512, 5122)
LOSS_FN = losses.huber_loss
ADAM_OPTIMIZER = optax.adam(learning_rate=0.001, eps=3.125e-4)
GAMMA = 0.99

REDUNDANCY = 3
STEPS = 1000
ITERATIONS = 500

ENV = gym.make("CartPole-v1")
ENV_PREPROC = constants.env_preproc_info("CartPole", "v1")["preproc"]
ENV_STATE_EXAMPLE = jnp.ones(constants.env_info(ENV)["observation_shape"] + (1,))


V_MODEL = networks.MLP(features=1, hiddens=MLP_HIDDEN_SIZE, **ENV_PREPROC)
V_ENSEMBLE = networks.EnsembledNet(model=V_MODEL, n_heads=ENSEMBLE_HEADS)

Q_MODEL = networks.MLP(
    features=ENV.action_space.n, hiddens=MLP_HIDDEN_SIZE, **ENV_PREPROC
)


def mark_trainable_params(
    head_idx: int,
    params: frozen_dict.FrozenDict,
    pos_mark: str = "sgd",
    neg_mark: str = "none",
) -> frozen_dict.FrozenDict:
    return frozen_dict.freeze(
        {
            "params": {
                h: pos_mark if h.endswith(f"_{head_idx}") else neg_mark
                for h in list(jax.tree_map(jnp.shape, params)["params"].keys())
            }
        }
    )


def make_v_model(rng: custom_pytrees.PRNGKeyWrap) -> List[custom_pytrees.DQVTrainState]:
    v_ensemble_params = V_ENSEMBLE.init(next(rng), ENV_STATE_EXAMPLE)
    return [
        custom_pytrees.DQVTrainState.create(
            apply_fn=ft.partial(V_ENSEMBLE.apply, head=i),
            params=v_ensemble_params,
            target_params=v_ensemble_params,
            tx=optax.multi_transform(
                {"adam": ADAM_OPTIMIZER, "none": optax.set_to_zero()},
                mark_trainable_params(i, v_ensemble_params),
            ),
            loss_metric=LOSS_FN,
        )
        for i in range(V_ENSEMBLE.n_heads)
    ]


def make_q_model(rng: custom_pytrees.PRNGKeyWrap) -> custom_pytrees.DQVTrainState:
    q_params = Q_MODEL.init(next(rng), ENV_STATE_EXAMPLE)
    return custom_pytrees.DQVTrainState.create(
        apply_fn=Q_MODEL.apply,
        params=q_params,
        target_params=None,
        tx=ADAM_OPTIMIZER,
        loss_metric=LOSS_FN,
    )


for repeat in range(REDUNDANCY):
    rng = custom_pytrees.PRNGKeyWrap(SEED)
    SEED += 1

    memory = offline_memory.OfflineOutOfGraphReplayBuffer(
        f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/{repeat}",
        load_parallel=False,
        **constants.env_info(ENV),
        **constants.default_memory_args,
    )

    q_train_state = make_q_model(rng)
    v_train_states = make_v_model(rng)

    for iteration in range(ITERATIONS):
        for t in range(STEPS):
            experience_batch = agent_utils.sample_replay_buffer(memory)
            td_targets = agent_utils.td_error(
                GAMMA,
                agent_utils.batch_net_eval(
                    V_ENSEMBLE.apply,
                    v_train_states[0].target_params,
                    experience_batch["next_state"],
                ).mean(axis=1),
                experience_batch["reward"],
                experience_batch["terminal"],
            )
