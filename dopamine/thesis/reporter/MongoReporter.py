import signal
from typing import List

import attr
import pymongo
from thesis.reporter import Reporter

# TODO report additional info on experiment: agent, environment etc!


@attr.s(auto_attribs=True)
class BufferedMongoCollection:
    buffering: int
    collection: pymongo.collection.Collection
    docs: List[dict] = attr.ib(factory=list, init=False)

    @property
    def size(self):
        return len(self.docs)

    def flush_docs(self):
        self.collection.insert_many(self.docs)
        self.docs = []

    def safe_flush_docs(self):
        if self.size:
            self.flush_docs()

    def __iadd__(self, doc: dict):
        self.docs.append(doc)
        if self.size >= self.buffering:
            self.flush_docs()
            print("Wrote buffered mongo docs")
        return self

    def flush_docs_handler(self, signum: int, frame):
        n_pending_docs = self.size
        self.safe_flush_docs()
        print(f"mongo handler on {signum}: Flushed {n_pending_docs} documents")
        raise KeyboardInterrupt

    def __repr__(self) -> str:
        return f"<{self.__class__}#docs:{self.size}>"


@attr.s(auto_attribs=True)
class MongoReporter(Reporter.Reporter):
    db_uri: str = "mongodb://localhost:27017"
    db_name: str = "thesis_db"
    collection_name: str = "thesis_collection"
    buffering: int = 50
    timeout: int = 30
    client: pymongo.MongoClient = attr.ib(init=False)
    db: pymongo.database.Database = attr.ib(init=False)
    collection: BufferedMongoCollection = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.client = pymongo.MongoClient(
            self.db_uri, serverSelectionTimeoutMS=self.timeout * 1000
        )
        self.db = self.client[self.db_name]
        self.collection = BufferedMongoCollection(
            self.buffering, self.db[self.collection_name]
        )
        signal.signal(signal.SIGINT, self.collection.flush_docs_handler)

    def __call__(self, raw_reports: dict, agg_reports: dict, runner_info: dict):
        self.collection += {
            "experiment": self.experiment_name,
            "run_hash": self.conf["run_hash"],
            **raw_reports,
            **runner_info,
        }


# naive method used to save arbitrary data to mongo by pickling them
# def save_checkpoint(self, record: dict, tag: str):
#     self.collection += {
#         **jax.tree_map(
#             lambda v: pickle.dumps(v)
#             if isinstance(v, (jax.numpy.DeviceArray, np.ndarray))
#             else v,
#             record,
#         ),
#         "collection_tag": tag,
#     }
