import gym
import optax
import tensorflow as tf
from dopamine.replay_memory import circular_replay_buffer
from torch import nn, optim

from thesis import dqn_jax, dqn_torch, memory, networks_torch, simple_runner

dqn_classic = ["SimpleDQN", "CartPole-v0", "SequentialMLP" "novision"]
dqn_classic_torch = ["SimpleDQN", "CartPole-v0", "SequentialMLP" "novision", "torch"]

data_dir = "/home/xqz-u/uni/thesis/dopamine/thesis/data_collection"
run_name = simple_runner.make_unique_data_dir(dqn_classic, data_dir)
run_name_torch = simple_runner.make_unique_data_dir(dqn_classic_torch, data_dir)
summary_writer = tf.summary.create_file_writer(run_name)
summary_writer_torch = tf.summary.create_file_writer(run_name_torch)

env = gym.make("CartPole-v0")
env.seed(0)

obs_shape, action_n = env.observation_space.shape, env.action_space.n
# obs_shape += (1,)

mem = circular_replay_buffer.OutOfGraphReplayBuffer(
    obs_shape, 1, 50000, 128, observation_dtype=env.observation_space.dtype
)
jax_q_net = dqn_jax.mlp([512, 512, 2])
jax_agent = dqn_jax.DQNAgent(
    action_n,
    obs_shape,
    memory=mem,
    q_net=jax_q_net,
    optim=optax.adam(learning_rate=0.001, eps=3.125e-4),
    summary_writer=summary_writer,
)


torch_q_net = networks_torch.simple_mlp([*obs_shape, 512, 512, action_n])
torch_mem = memory.ReplayMemory(50000, 128)
# torch_mem = memory.ReplayMemory(50000, 4)
torch_agent = dqn_torch.TorchDQN(
    action_n,
    torch_q_net,
    torch_mem,
    optim.Adam(torch_q_net.parameters()),
    nn.HuberLoss(reduction="mean"),
    target_update_freq=1000,
    summary_writer=summary_writer_torch,
)


simple_runner.run_exp(500, 1000, jax_agent, env)
# simple_runner.run_exp(500, 1000, torch_agent, env)
# simple_runner.run(torch_agent, env)


# import gin
# from dopamine.discrete_domains import run_experiment
# from dopamine.jax import networks

# gin.parse_config_file(
#     "/home/xqz-u/uni/thesis/dopamine/dopamine/jax/agents/dqn/configs/dqn_cartpole.gin"
# )
# dqn_dopamine = ["JaxDQNAgent", "CartPole-v0", "ClassicControlDQNNetwork", "ref", "1"]
# dqn_dopamine_run_name = simple_runner.make_unique_data_dir(dqn_dopamine, data_dir)
# runner = run_experiment.create_runner(dqn_dopamine_run_name)
# runner._agent.seed = 0
# runner._environment.environment.seed = 0
# runner.run_experiment()


# obs = env.reset()

# for _ in range(5):
#     print(obs)
#     action = torch_agent.select_action(obs)
#     obs, reward, done, _ = env.step(action)
#     torch_agent.learn(obs, reward, done)

# els = torch_agent.sample_memory()
