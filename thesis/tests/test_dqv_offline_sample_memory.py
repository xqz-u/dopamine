import os

import gin

import thesis.jax.agents.dqv_agent as dqv

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")

conf = os.path.join(THESIS, "tests", "test_dqv_offline_sample_memory.gin")

gin.parse_config_file(conf)
print(gin.operative_config_str())

agent = dqv.JaxDQVAgent()
dqv.build_networks(agent)
agent.networks_shape
x = agent.sample_memory()
# print(x)
