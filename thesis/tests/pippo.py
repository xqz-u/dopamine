import functools as ft
import time

import gin

import thesis.utils as u

# from thesis.tests.gin_multiple_calls import Pippo

# gin.parse_config_file(
#     "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine/thesis/tests/gin_multiple_calls.gin"
# )

# x = Pippo()
# print(x.mimmo)


class P:
    def __init__(self):
        self.store = []

    @u.timer
    def method(self, arg):
        time.sleep(2)
        print("done")
        print(arg)

    def reducer(self, arg0, arg):
        print(self)
        print(arg0)
        print(arg)
        self.store += [arg]
        return self

    def reduction(self):
        print(self.store)
        ft.reduce(self.reducer, range(5), self)
        print(self.store)


a = P()
# a.method(4)
a.reduction()
