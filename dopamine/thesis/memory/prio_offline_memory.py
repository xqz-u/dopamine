from thesis.memory import offline_memory


class PrioritizedOfflineOutOfGraphReplayBuffer(
    offline_memory.OfflineOutOfGraphReplayBuffer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_buffers(self, workers: int = None):
        ...
