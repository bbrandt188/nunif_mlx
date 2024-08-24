# mapper function to convert model output to disparity
# see also iw3/training/find_mapper.py
import mlx.core as mx
import math


def softplus01(x, bias, scale):
    # x: 0-1 normalized
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = mx.log(1. + mx.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def inv_softplus01(x, bias, scale):
    min_v = ((mx.zeros(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    max_v = ((mx.ones(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    v = ((x - bias) * scale).expm1().clamp(min=1e-6).log()
    return (v - min_v) / (max_v - min_v)


def distance_to_disparity(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def inv_distance_to_disparity(x, c):
    return ((c + 1) * x) / (x + c)


def shift(x, value):
    return (1.0 - value) * x + value


def div_shift(x, value, c=0.6):
    x = inv_distance_to_disparity(x, c)
    x = (1.0 - value) * x + value
    x = distance_to_disparity(x, c)
    return x


def resolve_mapper_function(name):
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01
    elif name == "softplus2":
        return lambda x: softplus01(x) ** 2
    elif name in {"mul_1", "mul_2", "mul_3"}:
        param = {
            "mul_1": {"bias": 0.343, "scale": 12},
            "mul_2": {"bias": 0.515, "scale": 12},
            "mul_3": {"bias": 0.687, "scale": 12},
        }[name]
        return lambda x: softplus01(x, **param)
    elif name in {"inv_mul_1", "inv_mul_2", "inv_mul_3"}:
        param = {
            "inv_mul_1": {"bias": -0.002102, "scale": 7.8788},
            "inv_mul_2": {"bias": -0.0003, "scale": 6.2626},
            "inv_mul_3": {"bias": -0.0001, "scale": 3.4343},
        }[name]
        return lambda x: inv_softplus01(x, **param)
    elif name in {"div_25", "div_10", "div_6", "div_4", "div_2", "div_1"}:
        param = {
            "div_25": 2.5,
            "div_10": 1,
            "div_6": 0.6,
            "div_4": 0.4,
            "div_2": 0.2,
            "div_1": 0.1,
        }[name]
        return lambda x: distance_to_disparity(x, param)
    elif name in {"shift_25", "shift_50", "shift_75"}:
        param = {
            "shift_25": 0.25,
            "shift_50": 0.5,
            "shift_75": 0.75,
        }[name]
        return lambda x: shift(x, param)
    elif name in {"div_shift_25", "div_shift_50", "div_shift_75"}:
        param = {
            "div_shift_25": 0.25,
            "div_shift_50": 0.5,
            "div_shift_75": 0.75,
        }[name]
        return lambda x: div_shift(x, param, 0.6)
    else:
        raise NotImplementedError(f"mapper={name}")


def chain(x, functions):
    for f in functions:
        x = f(x)
    return x


def get_mapper(name):
    if ":" in name:
        names = name.split(":")
    else:
        names = [name]
    functions = [resolve_mapper_function(name) for name in names]
    return lambda x: chain(x, functions)


def resolve_mapper_name(mapper, foreground_scale, metric_depth):
    disparity_mapper = None
    if mapper is not None:
        if mapper == "auto":
            if not metric_depth:
                disparity_mapper = "none"
            else:
                disparity_mapper = "div_6"
        else:
            disparity_mapper = mapper
    else:
        if not metric_depth:
            disparity_mapper = [
                "inv_mul_3", "inv_mul_2", "inv_mul_1",
                "none",
                "mul_1", "mul_2", "mul_3",
            ][foreground_scale + 3]
        else:
            disparity_mapper = [
                "none", "div_25", "div_10",
                "div_6",
                "div_4", "div_2", "div_1",
            ][foreground_scale + 3]

    assert disparity_mapper is not None
    return disparity_mapper
