import logging
from dataclasses import dataclass


class ConsoleLogger(logging.Logger):
    def __init__(self, level: int = logging.DEBUG, name: str = None, *args, **kwargs):
        super().__init__(f"{__name__}+{name}", *args, **kwargs)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
            datefmt="%m/%d/%Y %I:%M:%S",
        )
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.setLevel(level)


@dataclass
class A:
    level: int = logging.DEBUG

    def __post_init__(self):
        self.logger = ConsoleLogger(self.level, name="A")

    def pippo(self):
        for _ in range(5):
            self.logger.debug("debug")
            self.logger.info("info")


@dataclass
class B:
    level: int = logging.DEBUG

    def __post_init__(self):
        self.logger = ConsoleLogger(self.level, name="B")

    def pippo(self):
        for _ in range(5):
            self.logger.debug("debug B")
            self.logger.info("info B")


x = A(level=logging.INFO)
# x = A()
y = B()

x.pippo()
y.pippo()
