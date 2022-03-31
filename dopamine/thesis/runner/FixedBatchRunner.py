import operator
from collections import OrderedDict

import attr
from thesis import utils
from thesis.runner import Runner


# NOTE the way the offline_circular_replay_buffer works, this runner
# will load a new set of replay buffers for each redundancy,
# corresponding to the ones recorded online; if
# FixedBatchRunner.redundancy is different than the nr. of
# redundancies used to generate the data online, the runner will fail
# as it is
@attr.s(auto_attribs=True)
class FixedBatchRunner(Runner.Runner):
    def __attrs_post_init__(self):
        # we are not collecting new experiences at all, so force
        # fitting immediately
        self.conf["agent"]["min_replay_history"] = 0
        super().__attrs_post_init__()
        assert self.agent.min_replay_history == 0
        # TODO is this the right place for this call? move loading to
        # __init__ of OfflineOutOfGraphReplayBuffer...
        self.agent.memory.load_buffers()

    def train_iteration(self) -> OrderedDict:
        train_info = OrderedDict(
            loss=self.agent.init_loss(), steps=self.steps, q_estimates=0.0
        )
        for _ in range(self.steps):
            utils.inplace_dict_assoc(
                train_info, operator.add, update_dict=self.agent.learn()
            )
        self.global_steps += self.steps
        aggregate_info = Runner.aggregate_losses(self.agent.loss_names, train_info)
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    @property
    def console_name(self):
        return __name__
