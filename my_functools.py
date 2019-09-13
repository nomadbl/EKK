# import functools


def map2(f, x, y):
    return map(map(partial( map(lambda z: partial(f,z), x), y)


def zipwith(f, x, y):
    if len(x) != len(y):
        raise Exception("lists have different lengths")
    if len(x) == 1:
        return [f(x[0], y[0])]
    else:
        return [f(x[0], y[0])] + zipwith(f, x[1:], y[1:])
