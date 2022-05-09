import attr


@attr.s(auto_attribs=True)
class A:
    a: int = 2

    def __attrs_post_init__(self):
        print(f"A: {self.a}")


@attr.s(auto_attribs=True)
class B(A):
    b: int = 3

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        print(f"B: {self.b}")


x = B(a=4)


@attr.s(auto_attribs=True)
class C:
    a: int = 2

    def __attrs_post_init__(self):
        print(f"A: {self.a}")


@attr.s(auto_attribs=True)
class D(C):
    b: int = 3


x = D(a=4)
