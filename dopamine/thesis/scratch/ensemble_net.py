import functools as ft
from typing import Tuple

import gym
import jax
import optax
from dopamine.jax import losses
from flax.core import frozen_dict
from flax.training import train_state
from jax import numpy as jnp
from jax import random as jrand
from thesis import config, constants, custom_pytrees, networks
from thesis.agents import agent_utils
from thesis.memory import offline_memory


@ft.partial(jax.jit, static_argnums=(0,))
def test_valid_jaxtype(el):
    return el


mem = offline_memory.OfflineOutOfGraphReplayBuffer(
    f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
    [0],
    load_parallel=False,
    **constants.env_info(gym.make("CartPole-v1")),
    **constants.default_memory_args,
)
memory_sample = agent_utils.sample_replay_buffer(mem, batch_size=8)


rng = custom_pytrees.PRNGKeyWrap(0)
shape = (4, 1, 1)
init_arr = jnp.ones(shape)
test_arr = jrand.uniform(next(rng), shape)


cp_preproc = constants.env_preproc_info("CartPole", "v1")["preproc"]
# cp_preproc = {k: jnp.array(v) for k, v in cp_preproc.items()}
# model = networks.MLP(features=2, hiddens=(2, 4))
model = networks.MLP(features=1, hiddens=(2, 4), **cp_preproc)
# model = networks.DensePreproc(features=2, **cp_preproc)
model_params = model.init(next(rng), init_arr)
# print(jax.tree_map(jnp.shape, model_params))
# model.apply(model_params, test_arr)

# model_cp = model.clone()
# model_cp_params = model_cp.init(next(rng), init_arr)
# res = jnp.array(
#     [
#         m.apply(p, test_arr)
#         for m, p in zip([model, model_cp], [model_params, model_cp_params])
#     ]
# )
# res
# res.mean(axis=0)


# conv_shape = (84, 84, 4)
# conv_init = jnp.ones(conv_shape)
# conv_model = networks.NatureDQNNetwork(num_actions=2)
# conv_ens = networks.EnsembledNet(model=conv_model, n_heads=2)
# conv_ens_params = conv_ens.init(next(rng), conv_init)
# print(jax.tree_map(jnp.shape, conv_ens_params))


def make_adam_optim():
    optim = config.adam_huberloss["optim"]["call_"]
    return optim(
        **{k: v for k, v in config.adam_huberloss["optim"].items() if k != "call_"}
    )


v_model = networks.MLP(features=1, hiddens=(8,), **cp_preproc)
v_ens = networks.EnsembledNet(model=v_model, n_heads=2)
v_ens_params = v_ens.init(next(rng), init_arr)
optim = make_adam_optim()
dqv_tree = custom_pytrees.NetworkOptimWrap(
    v_ens_params, optim.init(v_ens_params), v_ens, optim
)
dqv_tree.params = dict(zip(["online", "target"], [dqv_tree.params] * 2))


# x = V(s_t+1; theta')
all_t1_targets = agent_utils.batch_net_eval(
    dqv_tree.net, dqv_tree.params["target"], memory_sample["next_state"]
)
# output is of the form (batch_size, ensamble_size, features), take
# the mean of the ensamble decision
t1_targets = all_t1_targets.mean(axis=1)
# y = r + gamma * x ## so the targets use a mean
dqv_td_error = agent_utils.td_error(
    0.99, t1_targets, memory_sample["reward"], memory_sample["terminal"]
)

# now train each network of the ensemble against the average target
t0_predictions = agent_utils.batch_net_eval(
    dqv_tree.net, dqv_tree.params["online"], memory_sample["state"]
)


@jax.jit
def train_step(
    state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray
) -> Tuple[jnp.ndarray, train_state.TrainState]:
    def loss_fn(params: frozen_dict.FrozenDict) -> jnp.ndarray:
        estimates = jax.vmap(lambda x: state.apply_fn(params, x))(inputs).squeeze()
        return jnp.mean(jax.vmap(losses.mse_loss)(targets, estimates))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, state.apply_gradients(grads=grads)


# example usage of a TrainState
state = train_state.TrainState.create(
    apply_fn=model.apply, params=model_params, tx=optax.sgd(0.1)
)
loss, new_state = train_step(state, memory_sample["state"], dqv_td_error)


def mark_trainable_params(
    head_idx: int, params: frozen_dict.FrozenDict
) -> frozen_dict.FrozenDict:
    return frozen_dict.freeze(
        {
            "params": {
                h: "sgd" if h.endswith(f"_{head_idx}") else "none"
                for h in list(jax.tree_map(jnp.shape, params)["params"].keys())
            }
        }
    )


# example of how to train each head of on ensemble NN against a common
# target
train_states = [
    train_state.TrainState.create(
        apply_fn=ft.partial(v_ens.apply, head=i),
        params=v_ens_params,
        tx=optax.multi_transform(
            {"sgd": optax.sgd(0.1), "none": optax.set_to_zero()},
            mark_trainable_params(i, v_ens_params),
        ),
    )
    for i in range(v_ens.n_heads)
]

for state in train_states:
    loss, new_state = train_step(state, memory_sample["state"], dqv_td_error)
    print(f"loss: {loss}\nNew state: {new_state}")


# TODO
# - check that networks creation works with the new ensembled net (I
#   mean agent_utils.build_net)
# - check whether the idea of having customizable training functions is
#   useful? in general yes, maybe it is unnecessary for my project?
#   could be safe to leave it for now...
