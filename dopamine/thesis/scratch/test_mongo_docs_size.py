import os
import pickle

from thesis import config, utils
from thesis.runner import runner
from thesis.scratch import test_mongo_reporter


def main():
    conf = test_mongo_reporter.make_config("test_mongo")
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    manager = runner.create_runner(conf)
    return manager


x = main()
agent = x.agent
mongo_rep = x.reporters["mongo"]
d = agent.bundle_and_checkpoint("", "")
# mongo_rep.save_checkpoint(d, "agent_data")
# mongo_rep.collection.safe_flush_docs()

# doc = mongo_rep.collection.collection.find_one({"collection_tag": "agent_data"})


# with open("pippo", "wb") as fd:
#     pickle.dump(d, fout)


ckpt_dir = os.path.join(
    config.data_dir,
    "CartPole-v0/DQVAgent/dqv_train/checkpoints/redundancy_0",
)
data_file = os.path.join(ckpt_dir, "ckpt.10")

# with open(data_file, "rb") as fd:
#     ag_data = pickle.load(fd)
agent.memory.load(ckpt_dir, 10)

save_elements = agent.memory._return_checkpointable_elements()
# with open("pippo", "wb") as fd:
#     pickle.dump(save_elements, fd)

# mongo_rep.save_checkpoint(save_elements, "memory_data")
# mongo_rep.collection.safe_flush_docs()
