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

    def __iadd__(self, doc: dict):
        self.docs.append(doc)
        if self.size >= self.buffering:
            self.coll.insert_many(self.docs)
            self.docs = []
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__}#docs:{self.size}>"


class MongoReporter(Reporter.Reporter):
    client: pymongo.MongoClient = attr.ib(init=False)
    db: pymongo.database.Database = attr.ib(init=False)
    collection: BufferedMongoCollection = attr.ib(init=False)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27107,
        db_name: str = "thesis_db",
        collection_name: str = "thesis_collection",
        buffering: int = 1000,
    ):
        self.client = pymongo.MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = BufferedMongoCollection(buffering, self.db[collection_name])

    def setup(self, *_, **__):
        ...

    def __call__(self, raw_reports: dict, agg_reports: dict, runner_info: dict):
        self.collection += {
            "experiment": self.experiment_name,
            **raw_reports,
            **runner_info,
        }
