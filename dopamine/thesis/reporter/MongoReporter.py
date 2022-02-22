import signal
from typing import List

import attr
import pymongo
from thesis.reporter import Reporter


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
        return self

    def flush_docs_handler(self, signum: int, frame):
        self.safe_flush_docs()
        print(f"mongo: Flushed {self.size} documents")
        raise KeyboardInterrupt

    def __repr__(self) -> str:
        return f"<{self.__class__}#docs:{self.size}>"


@attr.s(auto_attribs=True)
class MongoReporter(Reporter.Reporter):
    host: str = "localhost"
    port: int = 27017
    db_name: str = "thesis_db"
    collection_name: str = "thesis_collection"
    buffering: int = 100
    client: pymongo.MongoClient = attr.ib(init=False)
    db: pymongo.database.Database = attr.ib(init=False)
    collection: BufferedMongoCollection = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.client = pymongo.MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]
        self.collection = BufferedMongoCollection(
            self.buffering, self.db[self.collection_name]
        )
        signal.signal(signal.SIGINT, self.collection.flush_docs_handler)

    def setup(self, *_, **__):
        ...

    def __call__(self, raw_reports: dict, agg_reports: dict, runner_info: dict):
        self.collection += {
            "experiment": self.experiment_name,
            **raw_reports,
            **runner_info,
        }
