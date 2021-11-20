import gin


@gin.configurable
class Pippo:
    def __init__(self, mimmo: list):
        self.mimmo = mimmo

    @gin.register
    def add_mimmo(self, el):
        self.mimmo.append(el)


class Parent:
    def __init__(self):
        self.store = []

    def adder(self, el):
        self.store.append(el)


class Child(Parent):
    def __init__(self):
        self.store = [1]
        # NOTE not calling super's init

    def child_adder(self):
        self.adder(1)


a = Child()
a.child_adder()
print(a.store)
