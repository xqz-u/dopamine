from thesis.reporter.aim_reporter import AimReporter
from thesis.reporter.base import Reporter
from thesis.reporter.mongo_reporter import MongoReporter

AVAILABLE_REPORTERS = {"mongo": MongoReporter, "aim": AimReporter}
