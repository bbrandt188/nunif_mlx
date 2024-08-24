import mlx.functional as MF
import mlx.core as mx

def gaussian_blur(x):
    kernel = mx.tensor([
        [21, 31, 21],
        [31, 48, 31],
        [21, 31, 21],
    ], dtype=mx.float32, device=x.device).reshape(1, 1, 3, 3) / 256.0
    x = MF.pad(x, [1] * 4, mode="replicate")
    x = MF.conv2d(x, weight=kernel, bias=None, stride=1, padding=0, groups=1)
    return x

def dilate(x):
    return MF.max_pool2d(x, kernel_size=3, stride=1, padding=1)

def edge_weight(x):
    max_v = MF.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    min_v = MF.max_pool2d(x.neg(), kernel_size=3, stride=1, padding=1).neg()
    range_v = max_v.sub_(min_v)
    range_c = range_v.sub_(range_v.mean())
    range_s = range_c.pow(2).mean().add_(1e-6)
    w = mx.clamp(range_c.div_(range_s), -2, 2)
    w_min, w_max = w.min(), w.max()
    if w_max - w_min > 0:
        w = (w - w_min) / (w_max - w_min)
    else:
        w.fill_(0)

    return w

@mx.inference_mode()
def dilate_edge(x, n):
    for _ in range(n):
        w = edge_weight(x)
        x2 = gaussian_blur(x)
        x2 = dilate(x2)
        x = (x * (1 - w)) + (x2 * w)

    return x
