import os
from os import path
import pickle
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.device import create_device, autocast, device_is_mps # noqa
from nunif.models.data_parallel import DataParallelInference
from .dilation import dilate_edge

import mlx.core as mx
from mlx.transforms import functional as MF

HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
MODEL_FILES = {
    "ZoeD_N": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_N.pt"),
    "ZoeD_K": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_K.pt"),
    "ZoeD_NK": path.join(HUB_MODEL_DIR, "checkpoints", "ZoeD_M12_NK.pt"),
    # DepthAnything backbone
    "ZoeD_Any_N": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_metric_depth_indoor.pt"),
    "ZoeD_Any_K": path.join(HUB_MODEL_DIR, "checkpoints", "depth_anything_metric_depth_outdoor.pt"),
}
DEPTH_ANYTHING_MODELS = {"ZoeD_Any_N": "indoor", "ZoeD_Any_K": "outdoor"}


def get_name():
    return "ZoeDepth"

def load_model(model_type="ZoeD_N", gpu=0, height=None):
    assert model_type in MODEL_FILES
    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        try:
            if not os.getenv("IW3_DEBUG"):
                if model_type not in DEPTH_ANYTHING_MODELS:
                    model = mx.hub.load("nagadomi/ZoeDepth_iw3:main", model_type, config_mode="infer",
                                        pretrained=True, verbose=False, trust_repo=True)
                else:
                    model = mx.hub.load("nagadomi/Depth-Anything_iw3:main",
                                        "DepthAnythingMetricDepth",
                                        model_type=DEPTH_ANYTHING_MODELS[model_type], remove_prep=False,
                                        verbose=False, trust_repo=True)
            else:
                if model_type not in DEPTH_ANYTHING_MODELS:
                    assert path.exists("../ZoeDepth_iw3/hubconf.py")
                    model = mx.hub.load("../ZoeDepth_iw3", model_type, source="local", config_mode="infer",
                                        pretrained=True, verbose=False, trust_repo=True)
                else:
                    assert path.exists("../Depth-Anything_iw3/hubconf.py")
                    model = mx.hub.load("../Depth-Anything_iw3",
                                        "DepthAnythingMetricDepth",
                                        model_type=DEPTH_ANYTHING_MODELS[model_type], remove_prep=False,
                                        source="local", verbose=False, trust_repo=True)
        except (RuntimeError, pickle.PickleError) as e:
            if isinstance(e, RuntimeError):
                do_handle = "PytorchStreamReader" in repr(e)
            else:
                do_handle = True
            if do_handle:
                try:
                    # delete corrupted file
                    os.unlink(MODEL_FILES[model_type])
                except:  # noqa
                    pass
                raise RuntimeError(
                    f"File `{MODEL_FILES[model_type]}` is corrupted. "
                    "This error may occur when the network is unstable or the disk is full. "
                    "Try again."
                )
            else:
                raise

    # remove prep function if any
    model.core.prep = lambda x: x

    # Set up model-specific parameters
    if model_type not in DEPTH_ANYTHING_MODELS:
        model.prep_mod = 32
        if height is not None:
            if height % model.prep_mod != 0:
                height += (model.prep_mod - height % model.prep_mod)
            model.prep_h_height = height
            model.prep_v_height = height
        else:
            model.prep_h_height = 384
            model.prep_v_height = 512
    else:
        model.prep_mod = 14
        if height is not None:
            if height % model.prep_mod != 0:
                height += (model.prep_mod - height % model.prep_mod)
            model.prep_h_height = height
            model.prep_v_height = height
        else:
            model.prep_h_height = 392
            model.prep_v_height = 518

    # Move model to the appropriate device
    device = create_device(gpu)
    model = model.to(device).eval()
    model._model_type = model_type

    # Handle multi-GPU setups
    if isinstance(gpu, (list, tuple)) and len(gpu) > 1 and model_type not in DEPTH_ANYTHING_MODELS:
        model = DataParallelInference(model, device_ids=gpu)

    return model

def has_model(model_type="ZoeD_N"):
    assert model_type in MODEL_FILES
    return path.exists(MODEL_FILES[model_type])

def force_update_midas():
    with TorchHubDir(HUB_MODEL_DIR):
        mx.hub.help("nagadomi/MiDaS_iw3:master", "DPT_BEiT_L_384", force_reload=True, trust_repo=True)

def force_update_zoedepth():
    with TorchHubDir(HUB_MODEL_DIR):
        mx.hub.help("nagadomi/ZoeDepth_iw3:main", "ZoeD_N", force_reload=True, trust_repo=True)

def force_update_depth_anything():
    with TorchHubDir(HUB_MODEL_DIR):
        mx.hub.help("nagadomi/Depth-Anything_iw3:main", "DepthAnythingMetricDepth", force_reload=True, trust_repo=True)

def force_update():
    force_update_midas()
    force_update_zoedepth()
    force_update_depth_anything()

def _forward(model, x, enable_amp):
    with autocast(device=x.device, enabled=enable_amp)
        out = model(x)['metric_depth']
    out = mx.nan_to_num(out)
    return out
def batch_preprocess(x, h_height=384, v_height=512, ensure_multiple_of=32):
    # x: BCHW float32 0-1
    B, C, height, width = x.shape
    mod = ensure_multiple_of

    # resize + pad
    target_height = h_height if width > height else v_height
    if target_height < height:
        new_h = target_height
        new_w = int(new_h / height * width)
        if new_w % mod != 0:
            new_w += (mod - new_w % mod)
        if new_h % mod != 0:
            new_h += (mod - new_h % mod)
    else:
        new_h, new_w = height, width
        if new_w % mod != 0:
            new_w -= new_w % mod
        if new_h % mod != 0:
            new_h -= new_h % mod

    pad_src_h = int((height * 0.5) ** 0.5 * 3)
    pad_src_w = int((width * 0.5) ** 0.5 * 3)
    pad_scale_h = pad_src_h / (height + pad_src_h * 2)
    pad_scale_w = pad_src_w / (width + pad_src_w * 2)

    antialias = True

    if new_h > new_w:
        pad_h = round(new_h * pad_scale_h)
        frame_h = new_h - pad_h * 2
        frame_w = int(width * (frame_h / height))
        frame_w += frame_w % 2
        pad_w = (new_h - frame_w) // 2
        x = MF.interpolate(x, size=(frame_h, frame_w), mode="bilinear",
                           align_corners=False, antialias=antialias)
        x = MF.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")
    else:
        pad_h = round(new_h * pad_scale_h)
        pad_w = round(new_w * pad_scale_w)
        frame_h = new_h - pad_h * 2
        frame_w = new_w - pad_w * 2
        x = MF.interpolate(x, size=(frame_h, frame_w), mode="bilinear",
                           align_corners=False, antialias=antialias)
        x = MF.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")

    x = mx.clamp(x, 0, 1)

    # normalize
    mean = mx.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    stdv = mx.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    x = (x - mean) / stdv

    return x, pad_h, pad_w

@mx.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, int16=True, enable_amp=False,
                output_device="cpu", device=None, normalize_int16=False,
                edge_dilation=0, resize_depth=True, **kwargs):

    device = device if device is not None else model.device
    batch = False
    if mx.is_tensor(im):
        assert im.ndim == 3 or im.ndim == 4
        if im.ndim == 3:
            im = im.unsqueeze(0)
        else:
            batch = True
        x = im.to(device)
    else:
        # PIL
        x = MF.to_tensor(im).unsqueeze(0).to(device)

    org_size = x.shape[-2:]
    x, pad_h, pad_w = batch_preprocess(
        x,
        h_height=model.prep_h_height, v_height=model.prep_v_height,
        ensure_multiple_of=model.prep_mod)

    if not low_vram:
        if flip_aug:
            x = mx.cat([x, mx.flip(x, dims=[3])], dim=0)
        out = _forward(model, x, enable_amp)
    else:
        x_org = x
        out = _forward(model, x, enable_amp)
        if flip_aug:
            x = mx.flip(x_org, dims=[3])
            out2 = _forward(model, x, enable_amp)
            out = mx.cat([out, out2], dim=0)

    out = out[:, :, pad_h:-pad_h, pad_w:-pad_w]
    if edge_dilation > 0:
        out = -dilate_edge(-out, edge_dilation)
    if resize_depth and out.shape[-2:] != org_size:
        out = MF.interpolate(out, size=(org_size[0], org_size[1]),
                             mode="bilinear", align_corners=False)

    if flip_aug:
        if batch:
            n = out.shape[0] // 2
            z = mx.empty((n, *out.shape[1:]), device=out.device)
            for i in range(n):
                z[i] = (out[i] + mx.flip(out[i + n], dims=[2])) * 128
        else:
            z = (out[0:1] + mx.flip(out[1:2], dims=[3])) * 128
    else:
        z = out * 256
    if not batch:
        assert z.shape[0] == 1
        z = z.squeeze(0)

    if int16:
        if normalize_int16:
            max_v, min_v = z.max(), z.min()
            uint16_max = 0xffff
            if max_v - min_v > 0:
                z = uint16_max * ((z - min_v) / (max_v - min_v))
            else:
                z = mx.zeros_like(z)
        z = z.to(mx.int16)

    z = z.to(output_device)

    return z
