#!/usr/bin/env python

import signal
import time

import attr


@attr.s(auto_attribs=True)
class Myclass:
    docs: list = attr.ib(factory=list)

    def signal_handler(self, *args):
        print(args)
        raise KeyboardInterrupt

    def pippo(self):
        i = 0
        while True:
            time.sleep(2)
            self.docs.append(i)
            i += 1
            print(self.docs)


x = Myclass()
signal.signal(signal.SIGINT, x.signal_handler)
x.pippo()
