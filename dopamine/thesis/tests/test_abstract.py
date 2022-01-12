import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass

import attr


@dataclass
class Parent(ABC):
    pippo: int = 2

    @abstractmethod
    def ciccio(self):
        print("ciccio parent")


class Child(Parent):
    cip: str

    def ciccio(self):
        super().ciccio()
        print("ciccio child")

    def mimmo(self):
        print(f"pippo: {self.pippo} cip: {self.cip}")


# problem with dataclasses: no clean way to have required attributes
try:
    Child = dataclass(Child)
except TypeError as e:
    print(f"ERROR: {e}")

c = Child("cip")


# NOTE base class attributes are always resolved first also using
# @attr.s however, attributes with defaults are treated differently
# than in a @dataclass
@attr.s(auto_attribs=True)
class A(ABC):
    req: int
    x: int = 2

    @abstractmethod
    def ciccio(self):
        print(f"ciccio parent, req: {self.req} x: {self.x}")


inspect.signature(A)


# NOTE this works because B does not have new required arguments in
# derived class
@attr.s(auto_attribs=True)
class B(A):
    cip: str = "cip"

    def ciccio(self):
        super().ciccio()
        print(f"ciccio child, cip: {self.cip}")


# observe here
inspect.signature(B)

ex = B("pippi")
ex.ciccio()

# wrong if want to follow type annotations
ex = B(3, "pippi")
ex.ciccio()
# right
ex = B(3, cip="pippi")
ex.ciccio()

ex = B(cip="pippi", req=3)
ex.ciccio()


# NOTE this fails: there is a required attribute which does not have a
# correspondence in the base class, and since parent's attrs are
# resolved first, this results in invalid python:
# def __init__(req, x=2, new): ...
class C_wrong(A):
    new: str
    req: int = 1


try:
    C_wrong = attr.s(auto_attribs=True)(C_wrong)
except ValueError as e:
    print(f"attr error: {e}")


# NOTE solution: mark the required child attribute as kw_only, and
# look at what happens to __init__'s signature. In general, do the
# following:
# - define defaults in base class
# - if needed, override them in derived class
# - when derived class has new required attributes, use the pattern
#   below
# - remember that attributes are resolved in parent to child order
@attr.s(auto_attribs=True)
class C(A):
    req: int = 1
    new: str = attr.ib(kw_only=True)
    ne1: str = attr.ib(kw_only=True)


inspect.signature(C)


# NOTE when the base class defines defaults but a derived class fails to
# comply, the base default is overwritten!
@attr.s(auto_attribs=True)
class P:
    new: int = 2


@attr.s(auto_attribs=True)
class Q(P):
    new: int


inspect.signature(Q)


@attr.s(auto_attribs=True)
class MyBase:
    a: int
    b: str = "cicco"

    def fields(self):
        for a in attr.fields(type(self)):
            print(a)


@attr.s(auto_attribs=True)
class MyChild(MyBase):
    a: int = 2
    c: int = 4


inspect.signature(MyChild)

y = MyBase(1)
y.fields()
x = MyChild(b="pippo")
x.fields()


from thesis import utils

utils.attr_fields_d(x)
