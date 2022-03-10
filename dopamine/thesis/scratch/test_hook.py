def add_hook(hook: callable):
    def inner(fn: callable):
        def decorator(*fnargs, **fnkwargs):
            ret = fn(*fnargs, **fnkwargs)
            hook(ret)
            return ret

        return decorator

    return inner


def stu(arg):
    print(f"after {arg}")
    return 6


def ciccio(arg):
    print(f"after {arg}")
    return 2


@add_hook(stu)
@add_hook(ciccio)
def pippo():
    return 4


# pippo = add_hook(stu)(add_hook(ciccio)(pippo))

pippo()


class Cip:
    a: int = 4

    @add_hook(stu)
    def method(self):
        print("#method called")
        return self.a


# Cip.method = add_hook(stu)(Cip.method)
x = Cip()
x.method()
# x.method = add_hook(stu)(x.method)


import os

import gym
from dopamine.replay_memory import prioritized_replay_buffer
from thesis import config, constants, offline_circular_replay_buffer, utils
from thesis.agents import agent_utils
from thesis.runner import runner
from thesis.scratch import test_mongo_reporter

cartpole = gym.make("CartPole-v0")
cartpole_dict = constants.env_info("CartPole", "v0")
memory_args = {
    "observation_shape": cartpole_dict["observation_shape"],
    "stack_size": 1,
    "observation_dtype": cartpole.observation_space.dtype,
}


experiences_path = os.path.join(
    config.data_dir, "CartPole-v0/JaxDQNAgent/online_train/checkpoints"
)
off_memory = offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer(
    **memory_args, checkpoint_dir=experiences_path, iterations=[499]
)
off_memory.load_buffers()

minibatch = agent_utils.sample_replay_buffer(off_memory, batch_size=5)


trans_one = {k: minibatch[k][0] for k in minibatch.keys()}
trans_add = {
    k: v for k, v in trans_one.items() if k in ["state", "action", "reward", "terminal"]
}
trans_add["state"] = trans_add["state"].squeeze(1)

conf = test_mongo_reporter.make_config("test_exp_recorder")
conf["runner"]["exp_recorder"] = True
conf["memory"]["replay_capacity"] = 4
utils.data_dir_from_conf(conf["experiment_name"], conf)
run = runner.create_runner(conf)
# run.run_experiment_with_redundancy()


# run.agent.memory.add(*list(trans_add.values()))
