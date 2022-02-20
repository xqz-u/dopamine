def rec_op(keys, d, op, *args):
    k, *rest = keys
    if not rest:
        return op(d, k, *args)
    return rec_op(rest, d[k], op, *args)


setter = lambda d, k, v: d.__setitem__(k, v)
getter = lambda d, k, *_: d[k]


class Runner:
    data: dict
    # base_dir: str

    def __init__(self, **kwargs):
        self.data = dict(**kwargs)
        # self.base_dir = self[["runner", "base_dir"]]

    def __getitem__(self, keys):
        # print(len(keys))
        # print(keys)
        return rec_op(keys, self.data, getter)

    def __setitem__(self, keys, value):
        return rec_op(keys, self.data, setter, value)

    def __repr__(self):
        return repr(self.data)

    # def star(self, *keys):
    #     print(keys, len(keys))


c = {"runner": {"base_dir": "mona", "mimmo": {"cip": 2}}, "a": 2}

r = Runner(**c)
r


# NOTE wanted: r["runner", "mimmo", "fck"], but
# cannot differentiate between cases where r[("a", "b")] is one single
# key and r["a", "b"] is two different keys of nested dictionaries.
# same happens if __getitem__ takes *keys, there is just one additional
# layer of nesting
r[["runner", "mimmo", "fck"]] = 42
r
r[["runner", "mimmo", "fck"]] = 43
r
r[["a"]] = "eucor"
r
r[["a"]]


from collections.abc import Mapping


class Runner1(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update(
            {k: Runner1(**v) for k, v in kwargs.items() if isinstance(v, Mapping)}
        )

    def __getitem__(self, keys):
        if isinstance(keys, str):
            return super().__getitem__(keys)
        for x in keys:
            self = self[x]
        return self

    def __setitem__(self, keys, value):
        value = Runner1(**value) if isinstance(value, Mapping) else value
        if isinstance(keys, str):
            return super().__setitem__(keys, value)
        return super(Runner1, self[keys[:-1]]).__setitem__(keys[-1], value)


def test1():
    return Runner1(**c)


def test12():
    r1 = Runner1(**c)
    r1["runner", "mimmo", "fck"] = 42
    r1["runner", "mimmo", "fck"] = 43
    r1["a"] = "eucor"
    r1["a"]
    return r1
