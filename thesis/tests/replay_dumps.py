import os

import gin

from thesis import dqn_agent_replay as rep_dqn

# from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")
checkpoints_dir = os.path.join(THESIS, "checkpoints")
dqn_cartpole_replay_conf = os.path.join(THESIS, "dqn_cartpole_replay.gin")


@gin.configurable
def cartpole_dqn_buf_loader(buf):
    buf.load(checkpoints_dir, "49")
    return buf


gin.parse_config_file(dqn_cartpole_replay_conf)

dqn = rep_dqn.JaxDQNAgent()
