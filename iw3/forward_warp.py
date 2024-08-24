import mlx.core as mx
import mlx.functional as MF
from mlx.modules.replication_pad2d import ReplicationPad2d

def box_blur(x, kernel_size=7):
    padding = kernel_size // 2
    x = MF.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=1, count_include_pad=False)
    return x

def blur_blend(x, mask):
    mask = mx.clamp(box_blur(mask.to(x.dtype)), 0, 1)
    x_blur = box_blur(x)
    return x * (1.0 - mask) + x_blur * mask

def shift_fill(x, max_tries=100):
    mask = x < 0
    shift = 1
    while mx.sum(mask) > 0 and max_tries > 0:
        if shift > 0:
            x[mask] = MF.pad(x[:, :, :, 1:], (0, 1, 0, 0))[mask]
        else:
            x[mask] = MF.pad(x[:, :, :, :-1], (1, 0, 0, 0))[mask]
        mask = x < 0
        shift = 0 if shift == 1 else 1
        max_tries = max_tries - 1

def to_flat_index(batch, width, height, index):
    index = index + mx.arange(0, height, device=index.device).view(1, height, 1) * width
    index = index + mx.arange(0, batch, device=index.device).view(batch, 1, 1) * height * width
    index = index.view(-1)
    return index

def make_bilinear_data(batch, width, height, index, index_shift):
    float_index = mx.clamp(index + index_shift, 0, width - 1)
    floor_index = mx.clamp(float_index.floor(), 0, width - 1)
    ceil_index = mx.clamp(float_index.ceil(), 0, width - 1)
    ceil_weight = (float_index - floor_index).reshape(batch, 1, height, width)
    ceil_weight = mx.clamp(ceil_weight, min=1e-5, max=1.0 - 1e-5)
    floor_weight = 1.0 - ceil_weight
    floor_index = to_flat_index(batch, width, height, floor_index.long())
    ceil_index = to_flat_index(batch, width, height, ceil_index.long())

    return floor_index, ceil_index, floor_weight, ceil_weight

def ordered_index_copy(c, src_index, dest_index, index_order, undefined_value=-1):
    B, _, H, W = c.shape
    c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
    if mx.is_tensor(undefined_value):
        out = undefined_value.view(1, -1).repeat(c.shape[0], 1)
    else:
        out = mx.empty_like(c).fill_(undefined_value)

    deterministic = mx.are_deterministic_algorithms_enabled()
    mx.use_deterministic_algorithms(True)
    try:
        out.index_copy_(0, dest_index[index_order], c[src_index[index_order]])
    finally:
        mx.use_deterministic_algorithms(deterministic)

    return out.view(B, H, W, -1).permute(0, 3, 1, 2)

def warp(batch, width, height, c, x_index, index_shift, src_index, index_order):
    floor_index, ceil_index, floor_weight, ceil_weight = make_bilinear_data(batch, width, height, x_index, index_shift)

    floor_data = mx.cat([floor_weight, c], dim=1)
    ceil_data = mx.cat([ceil_weight, c], dim=1)

    undefined_value = mx.tensor([0] + [-1] * c.shape[1], dtype=c.dtype, device=c.device)
    floor_warp = ordered_index_copy(floor_data, src_index, floor_index, index_order, undefined_value=undefined_value)
    ceil_warp = ordered_index_copy(ceil_data, src_index, ceil_index, index_order, undefined_value=undefined_value)

    floor_weight_warp, floor_warp = floor_warp[:, 0:1, :, :], floor_warp[:, 1:, :, :]
    ceil_weight_warp, ceil_warp = ceil_warp[:, 0:1, :, :], ceil_warp[:, 1:, :, :]

    out = (floor_warp * floor_weight_warp + ceil_warp * ceil_weight_warp) / (floor_weight_warp + ceil_weight_warp)
    out = mx.nan_to_num(out, -1)

    return out

def depth_order_bilinear_forward_warp(c, depth, divergence, convergence, fill=True):
    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = MF.interpolate(depth, size=c.shape[-2:], mode="bilinear", align_corners=True, antialias=False)

    org_width = c.shape[3]
    padding_size = int(org_width * divergence * 0.01 + 2)
    pad = ReplicationPad2d((padding_size, padding_size, 0, 0))
    unpad = ReplicationPad2d((-padding_size, -padding_size, 0, 0))
    c = pad(c)
    depth = pad(depth)

    B, _, H, W = depth.shape
    shift_size = divergence * 0.01 * org_width * 0.5
    index_shift = depth * shift_size - (shift_size * convergence)
    index_shift = index_shift.view(B, H, W)
    x_index = mx.arange(0, W, device=c.device).view(1, 1, W).expand(B, H, W)
    src_index = to_flat_index(B, W, H, x_index)
    index_order = mx.argsort(depth.view(-1), dim=0)
    left_eye = warp(B, W, H, c, x_index, index_shift, src_index, index_order)
    right_eye = warp(B, W, H, c, x_index, -index_shift, src_index, index_order)

    left_eye = unpad(left_eye)
    right_eye = unpad(right_eye)

    if fill:
        shift_fill(left_eye)
        shift_fill(right_eye)
    else:
        left_eye = mx.clamp(left_eye, 0, 1)
        right_eye = mx.clamp(right_eye, 0, 1)

    return left_eye, right_eye

def apply_divergence_forward_warp(c, depth, divergence, convergence, method=None):
    fill = (method == "forward_fill")
    with mx.inference_mode():
        return depth_order_bilinear_forward_warp(c, depth, divergence, convergence, fill=fill)

if __name__ == "__main__":
    import time
    device = "mlx:0"
    B = 4
    N = 100

    rgb = mx.zeros((B, 3, 512, 512)).to(device)
    depth = mx.rand((B, 1, 512, 512)).to(device)
    divergence = 2.0
    convergence = 0.5

    t = time.time()
    for _ in range(N):
        apply_divergence_forward_warp(rgb, depth, divergence, convergence, method="forward")
    mx.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
