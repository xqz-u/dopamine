import gin
from attrs import define

gin.enter_interactive_mode()


@gin.configurable
@define
class Inner:
    dims: tuple = (1, 1)

    def print(self):
        print(self.dims)


@gin.configurable
@define
class Outer:
    in_field: Inner
    num: int

    def tell(self):
        self.in_field.print()
        print(self.num)


bindings = """
Inner.dims = (2, 2)

Outer.in_field = @Inner()
Outer.num = int(1e6)
"""

gin.parse_config(bindings)

x = Outer()

x.tell()


# @gin.configurable
@define
class Parent:
    parent_field: int = 2


@gin.configurable
@define
class Child(Parent):
    child_field: int = 3


inheritance_binds = """
Child.parent_field = 4
Child.child_field = "a"
"""

gin.parse_config(inheritance_binds)


c = Child()
c.parent_field
c.child_field
