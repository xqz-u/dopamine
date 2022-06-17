import logging
import signal
from typing import List

import gin
import pymongo
from attrs import define, field
from thesis import types
from thesis.reporter import base

logger = logging.getLogger(__name__)


@define
class BufferedMongoCollection:
    buffering: int
    collection: pymongo.collection.Collection
    docs: List[dict] = field(init=False, factory=list)

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
            logger.debug("Wrote buffered Mongo docs")
        return self

    def flush_docs_handler(self, signum: int, frame):
        n_pending_docs = self.size
        self.safe_flush_docs()
        logger.warning(f"Mongo: handle {signum} flushing {n_pending_docs} documents")
        raise KeyboardInterrupt

    def __repr__(self) -> str:
        return f"<{self.__class__}#docs:{self.size}>"


# NOTE experimnt_name determines the name for the `metrics_collection`
# and `configs_collection` database
@gin.configurable
@define
class MongoReporter(base.Reporter):
    db_uri: str = "mongodb://localhost:27017"
    metrics_collection: str = "metrics"
    configs_collection: str = "configs"
    metrics_buffering: int = 50
    timeout: int = 30
    client: pymongo.MongoClient = field(init=False)
    db: pymongo.database.Database = field(init=False)
    collection: BufferedMongoCollection = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.client = pymongo.MongoClient(
            self.db_uri, serverSelectionTimeoutMS=self.timeout * 1000
        )
        logger.info(f"Opened Mongo client connection at {self.db_uri}")
        self.db = self.client[self.experiment_name]
        self.collection = BufferedMongoCollection(
            self.metrics_buffering, self.db[self.metrics_collection]
        )
        logger.info(
            f"Writing on db: {self.experiment_name} collection: {self.metrics_collection}"
        )
        # dump any pending metrics on Cc-Cc
        signal.signal(signal.SIGINT, self.collection.flush_docs_handler)

    def __call__(self, _, summ_reports: types.MetricsDict, experiment_info: dict):
        self.collection += {
            **summ_reports,
            **experiment_info,
        }

    def finalize(self):
        self.collection.safe_flush_docs()
        logger.info("Flushed any pending Mongo metrics documents")

    # write the experiment configuration to the `configs` database
    # under collection `collection_name`, once
    def register_conf(self, conf: dict):
        logger.info("Registering experiment config in Mongo...")
        configs_coll = self.client[self.experiment_name][self.configs_collection]
        logger.info(f"db: {self.experiment_name} collection: {self.configs_collection}")
        # mongo inserts a top-level _id key in documents
        configs_coll.insert_one(conf.copy())
