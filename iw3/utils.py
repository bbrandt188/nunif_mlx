import os
from os import path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import threading
import math
from tqdm import tqdm
from PIL import ImageDraw, Image
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.models import load_model  # , compile_model
import nunif.utils.video as VU
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir, TorchHubDir
from nunif.device import create_device, autocast, device_is_mps
from . import export_config
from . dilation import dilate_edge
from . forward_warp import apply_divergence_forward_warp
from . anaglyph import apply_anaglyph_redcyan
from . mapper import get_mapper, resolve_mapper_name

import mlx.core as mx
import math

HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
REMBG_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "rembg")
os.environ["U2NET_HOME"] = path.abspath(path.normpath(REMBG_MODEL_DIR))

ROW_FLOW_V2_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v2_20240130.pth"
ROW_FLOW_V3_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_20240423.pth"
ROW_FLOW_V3_SYM_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_sym_20240424.pth"

IMAGE_IO_QUEUE_MAX = 100


def normalize_depth(depth, depth_min=None, depth_max=None):
    # depth = depth.float()
    depth = depth.astype(mx.float32) # Convert to float
    
    if depth_min is None:
        depth_min = mx.min(depth)
        depth_max = mx.max(depth)
        # depth_min = depth.min()
        # depth_max = depth.max()

    if depth_max - depth_min > 0:
        depth = 1.0 - ((depth - depth_min) / (depth_max - depth_min))
        # depth = 1. - ((depth - depth_min) / (depth_max - depth_min))
    else:
        depth = mx.zeros_like(depth)
        # depth = torch.zeros_like(depth)
    # return torch.clamp(depth, 0., 1.)
    return mx.clip(depth, 0.0, 1.0)
    

def convert_normalized_depth_to_uint16_numpy(depth):
    uint16_max = 0xffff
    depth = uint16_max * depth
    depth = depth.astype(mx.init16).numpy().astype(np.uint16)
    # depth = depth.to(torch.int16).numpy().astype(np.uint16)
    return depth


def make_divergence_feature_value(divergence, convergence, image_width):
    # assert image_width <= 2048
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0

    return divergence_feature_value, convergence_feature_value


def make_input_tensor(c, depth, divergence, convergence,
                      image_width, depth_min=None, depth_max=None,
                      mapper="pow2", normalize=True):
    if normalize:
        depth = normalize_depth(depth.squeeze(0), depth_min, depth_max)
    else:
        depth = depth.squeeze(0)  # CHW -> HW
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = mx.full_like(depth, divergence_value)
    convergence_feat = mx.full_like(depth, convergence_value)
    # divergence_feat = torch.full_like(depth, divergence_value, device=depth.device)
    # convergence_feat = torch.full_like(depth, convergence_value, device=depth.device)

    if c is not None:
        w, h = c.shape[2], c.shape[1]
        mesh_y, mesh_x = mx.meshgrid(mx.linspace(-1, 1, h),
                                     mx.linspace(-1, 1, w), indexing="ij")
        grid = mx.stack((mesh_x, mesh_y), axis=2)
        grid = grid.permute(2, 0, 1)  # CHW
        
        return mx.concatenate([
            c,
            depth.[None, :, :],
            divergence_feat.[None, :, :],
            convergence_feat.[None, :, :],
            grid,
        ], axis=0)
    else:
        return mx.concatenate([
            depth.[None, :, :],
            divergence_feat.[None, :, :],
            convergence_feat.[None, :, :],
        ], axis=0)


def equirectangular_projection(c, device="cpu"):
    c = mx.to_device(c, device)
    h, w = c.shape[1:]
    max_edge = max(h, w)
    output_size = max_edge + max_edge // 2
    pad_w = (output_size - w) // 2
    pad_h = (output_size - h) // 2
    
    c = mx.pad(c, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant", value=0)

    h, w = c.shape[1:]
    y, x = mx.meshgrid(mx.linspace(-1, 1, h), mx.linspace(-1, 1, w), indexing="ij")

    azimuth = x * (math.pi * 0.5)
    elevation = y * (math.pi * 0.5)
    mesh_x = (max_edge / output_size) * mx.tan(azimuth)
    mesh_y = (max_edge / output_size) * (mx.tan(elevation) / mx.cos(azimuth))
    grid = mx.stack((mesh_x, mesh_y), axis=2)

    # Assuming `device_is_mps` and `F.grid_sample` equivalents are implemented in MLX
    if device_is_mps(c.device):
        # MPS does not support bicubic
        mode = "bilinear"
    else:
        mode = "bicubic"

    z = mx.grid_sample(c[None, :, :, :],
                      grid[None, :, :, :],
                      mode=mode, padding_mode="zeros",
                      align_corners=True)
    z = mx.clip(z.queeze(0), 0, 1)

    return z


def backward_warp(c, grid, delta, delta_scale):
    # Adjust the grid with delta and delta_scale
    grid = grid + delta * delta_scale
    # resize grid if necessary
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = mx.interpolate(grid, size=c.shape[-2:], mode="bilinear", align_corners=True, antialias=False)
    # Permute the grid dimensions to match MLX's expected input
    grid = grid.transpose(0, 2, 3, 1) # Equivalent to permute in pytorch

    # Set interpolation mode and padding based on the device
    if device_is_mps(c.device):
        # MPS does not support bicubic and border
        mode = "bilinear"
        padding_mode = "reflection"
    else:
        mode = "bicubic"
        padding_mode = "border"

    # Perform grid sampling
    z = mx.grid_sample(c, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    # Clamp the values between 0 and 1
    z = mx.clip(z, 0, 1)
    return z


def make_grid(batch, width, height, device):
    # Create Meshgrid
    mesh_y, mesh_x = mx.meshgrid(mx.linspace(-1, 1, height, device=device),
                                 mx.linspace(-1, 1, width, device=device), 
                                 indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).broadcast_to((batch, 1, height, width))
    mesh_x = mesh_x.reshape(1, 1, height, width).broadcast_to((batch, 1, height, width))
    grid = mx.concatenate((mesh_x, mesh_y), axis=1)
    return grid


def apply_divergence_grid_sample(c, depth, divergence, convergence):
    # BCHW
    B, _, H, W = depth.shape
    shift_size = divergence * 0.01
    index_shift = depth * shift_size - (shift_size * convergence)
    delta = mx.concatenate([index_shift, mx.zeros_like(index_shift)], axis=1)
    grid = make_grid(B, W, H, c.device)
    left_eye = backward_warp(c, grid, -delta, 1)
    right_eye = backward_warp(c, grid, delta, 1)

    return left_eye, right_eye


def apply_divergence_nn_LR(model, c, depth, divergence, convergence,
                           mapper, enable_amp):
    if getattr(model, "symmetric", False):
        left_eye, right_eye = apply_divergence_nn_symmetric(model, c, depth, divergence, convergence, mapper, enable_amp)
    else:
        left_eye = apply_divergence_nn(model, c, depth, divergence, convergence,
                                       mapper, -1, enable_amp)
        right_eye = apply_divergence_nn(model, c, depth, divergence, convergence,
                                        mapper, 1, enable_amp)
    return left_eye, right_eye


def apply_divergence_nn(model, c, depth, divergence, convergence,
                        mapper, shift, enable_amp):
    # BCHW
    assert model.delta_output

    # Flip the image if shift is positive
    if shift > 0:
        c = mx.flip(c, axis=3)
        depth = mx.flip(depth, axis=3)

    B, _, H, W = depth.shape
                            
    x = mx.stack([make_input_tensor(None, depth[i],
                                       divergence=divergence,
                                       convergence=convergence,
                                       image_width=W,
                                       mapper=mapper,
                                       normalize=False)  # already normalized
                     for i in range(depth.shape[0])])
    # Use autocast if available in MLX
    with autocast(device=depth.device, enabled=enable_amp):
        delta = model(x)
        
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    z = backward_warp(c, grid, delta, delta_scale)

    if shift > 0:
        z = mx.flip(z, axis=3)

    return z


def apply_divergence_nn_symmetric(model, c, depth, divergence, convergence,
                                  mapper, enable_amp):
    # BCHW
    assert model.delta_output
    assert model.symmetric
    B, _, H, W = depth.shape

    # Stack the input tensors
    x = mx.stack([make_input_tensor(None, depth[i],
                                    divergence=divergence,
                                    convergence=convergence,
                                    image_width=W,
                                    mapper=mapper,
                                    normalize=False)  # already normalized
                  for i in range(depth.shape[0])])

    # Use autocast if available in MLX
    with autocast(device=depth.device, enabled=enable_amp):
        delta = model(x)

    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    
    # Apply backward warp for both eyes
    left_eye = backward_warp(c, grid, delta, delta_scale)
    right_eye = backward_warp(c, grid, -delta, delta_scale)

    return left_eye, right_eye

def has_rembg_model(model_type):
    return path.exists(path.join(REMBG_MODEL_DIR, f"{model_type}.onnx"))


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF_Full_SBS"
HALF_SBS_SUFFIX = "_LR"
FULL_TB_SUFFIX = "_TBF_fulltb"
HALF_TB_SUFFIX = "_TB"

VR180_SUFFIX = "_180x180_LR"
ANAGLYPH_SUFFIX = "_redcyan"
DEBUG_SUFFIX = "_debug"

# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def make_output_filename(input_filename, args, video=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if args.vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif args.half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
    elif args.tb:
        auto_detect_suffix = FULL_TB_SUFFIX
    elif args.half_tb:
        auto_detect_suffix = HALF_TB_SUFFIX
    elif args.anaglyph:
        auto_detect_suffix = ANAGLYPH_SUFFIX + f"_{args.anaglyph}"
    elif args.debug_depth:
        auto_detect_suffix = DEBUG_SUFFIX
    else:
        auto_detect_suffix = FULL_SBS_SUFFIX

    def to_deciaml(f, scale, zfill=0):
        s = str(int(f * scale))
        if zfill:
            s = s.zfill(zfill)
        return s

    if args.metadata == "filename":
        if args.zoed_height:
            resolution = f"{args.zoed_height}_"
        else:
            resolution = ""
        if args.tta:
            tta = "TTA_"
        else:
            tta = ""
        if args.ema_normalize and video:
            ema = f"_ema{to_deciaml(args.ema_decay, 100, 2)}"
        else:
            ema = ""
        metadata = (f"_{args.depth_model}_{resolution}{tta}{args.method}_"
                    f"d{to_deciaml(args.divergence, 10, 2)}_c{to_deciaml(args.convergence, 10, 2)}_"
                    f"di{args.edge_dilation}_fs{args.foreground_scale}_ipd{to_deciaml(args.ipd_offset, 1)}{ema}")
    else:
        metadata = ""

    return basename + metadata + auto_detect_suffix + (args.video_extension if video else ".png")


def save_image(im, output_filename):
    im.save(output_filename)


def remove_bg_from_image(im, bg_session):
    # Perform background removal, ensuring higher mask resolution if possible
    mask = TF.to_tensor(rembg.remove(im, session=bg_session, only_mask=True, post_process_mask=True))

    # Convert image to MLX tensor
    im = mx.array(im).astype(mx.float32) / 255.0  # Normalize to [0, 1] range
    mask = mx.array(mask).astype(mx.float32)

    # Define background color
    bg_color = mx.array((0.4, 0.4, 0.2)).reshape(3, 1, 1)

    # Apply mask
    im = im * mask + bg_color * (1.0 - mask)
    
    # Clamp the values between 0 and 1
    im = mx.clip(im, 0, 1)
    
    # Convert back to PIL Image
    im = Image.fromarray((im * 255).astype("uint8"))

    return im

def preprocess_image(im, args):
    # Convert image to MLX tensor if it is not already a tensor
    if not isinstance(im, mx.array):
        im = mx.array(im).astype(mx.float32) / 255.0  # Normalize to [0, 1]

    # Rotate the image if specified
    if args.rotate_left:
        im = mx.rot90(im, k=1, axes=(1, 2))
    elif args.rotate_right:
        im = mx.rot90(im, k=3, axes=(1, 2))

    h, w = im.shape[1:]
    new_w, new_h = w, h

    # Resize the image if it exceeds the maximum output height
    if args.max_output_height is not None and new_h > args.max_output_height:
        new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height

    # Ensure dimensions are even
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        im = mx.resize(im, (new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        im = mx.clip(im, 0, 1)

    im_org = im

    # Apply background removal if bg_session is provided
    if args.bg_session is not None:
        im2 = remove_bg_from_image(Image.fromarray((im * 255).astype("uint8").transpose(1, 2, 0)), args.bg_session)
        im = mx.array(im2).astype(mx.float32) / 255.0  # Convert back to tensor and normalize

    return im_org, im

def apply_divergence(depth, im_org, args, side_model, ema=False):
    batch = True
    if depth.ndim != 4:
        # CHW
        depth = depth[None, ...]  # Add batch dimension
        im_org = im_org[None, ...]
        batch = False

    # Normalize depth values
    for i in range(depth.shape[0]):
        depth_min, depth_max = mx.min(depth[i]), mx.max(depth[i])
        if ema:
            depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)
        depth[i] = normalize_depth(depth[i], depth_min=depth_min, depth_max=depth_max)

    # Apply divergence based on the selected method
    if args.method in {"grid_sample", "backward"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_grid_sample(
            im_org, depth, args.divergence, convergence=args.convergence)
    elif args.method in {"forward", "forward_fill"}:
        depth = get_mapper(args.mapper)(depth)
        left_eye, right_eye = apply_divergence_forward_warp(
            im_org, depth, args.divergence, convergence=args.convergence,
            method=args.method)
    else:
        if args.stereo_width is not None:
            # Rescale the depth if stereo width is specified
            H, W = im_org.shape[2:]
            stereo_width = min(W, args.stereo_width)
            if depth.shape[3] != stereo_width:
                new_w = stereo_width
                new_h = int(H * (stereo_width / W))
                depth = mx.resize(depth, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True)
                depth = mx.clip(depth, 0, 1)
        left_eye, right_eye = apply_divergence_nn_LR(
            side_model, im_org, depth, args.divergence, args.convergence,
            mapper=args.mapper, enable_amp=not args.disable_amp)

    # Remove the batch dimension if it was added
    if not batch:
        left_eye = left_eye[0, ...]
        right_eye = right_eye[0, ...]

    return left_eye, right_eye

def postprocess_image(left_eye, right_eye, args):
    # CHW
    ipd_pad = int(abs(args.ipd_offset) * 0.01 * left_eye.shape[2])
    ipd_pad -= ipd_pad % 2

    if ipd_pad > 0:
        pad_o, pad_i = (ipd_pad * 2, ipd_pad) if args.ipd_offset > 0 else (ipd_pad, ipd_pad * 2)
        left_eye = mx.pad(left_eye, (pad_o, 0, pad_i, 0), mode="constant")
        right_eye = mx.pad(right_eye, (pad_i, 0, pad_o, 0), mode="constant")

    if args.pad is not None:
        pad_h = int(left_eye.shape[1] * args.pad) // 2
        pad_w = int(left_eye.shape[2] * args.pad) // 2
        left_eye = mx.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), mode="constant")
        right_eye = mx.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), mode="constant")

    if args.vr180:
        left_eye = equirectangular_projection(left_eye, device=args.state["device"])
        right_eye = equirectangular_projection(right_eye, device=args.state["device"])
    elif args.half_sbs:
        left_eye = mx.resize(left_eye, (left_eye.shape[1], left_eye.shape[2] // 2), mode="bicubic", align_corners=False)
        right_eye = mx.resize(right_eye, (right_eye.shape[1], right_eye.shape[2] // 2), mode="bicubic", align_corners=False)
    elif args.half_tb:
        left_eye = mx.resize(left_eye, (left_eye.shape[1] // 2, left_eye.shape[2]), mode="bicubic", align_corners=False)
        right_eye = mx.resize(right_eye, (right_eye.shape[1] // 2, right_eye.shape[2]), mode="bicubic", align_corners=False)

    if args.anaglyph is not None:
        # Anaglyph
        sbs = apply_anaglyph_redcyan(left_eye, right_eye, args.anaglyph)
    elif args.tb or args.half_tb:
        # TopBottom
        sbs = mx.concatenate([left_eye, right_eye], axis=1)
    else:
        # SideBySide
        sbs = mx.concatenate([left_eye, right_eye], axis=2)

    sbs = mx.clip(sbs, 0.0, 1.0)

    h, w = sbs.shape[1:]
    new_w, new_h = w, h

    if args.max_output_height is not None and new_h > args.max_output_height:
        if args.keep_aspect_ratio:
            new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height

    if args.max_output_width is not None and new_w > args.max_output_width:
        if args.keep_aspect_ratio:
            new_h = int(args.max_output_width / new_w * new_h)
        new_w = args.max_output_width

    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        sbs = mx.resize(sbs, (new_h, new_w), mode="bicubic", align_corners=False)
        sbs = mx.clip(sbs, 0.0, 1.0)

    return sbs

def debug_depth_image(depth, args, ema=False):
    depth = depth.astype(mx.float32)  # Convert to float32 if not already
    depth_min, depth_max = mx.min(depth), mx.max(depth)

    if ema:
        depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)

    mean_depth, std_depth = float(mx.mean(depth)), float(mx.std(depth))

    # Normalize the depth image
    depth = normalize_depth(depth, depth_min=depth_min, depth_max=depth_max)

    # Apply the mapper to the depth image
    depth2 = get_mapper(args.mapper)(depth)

    # Concatenate depth images along the width dimension
    out = mx.concatenate([depth, depth2], axis=2).cpu()

    # Convert to PIL Image
    out = Image.fromarray((out.numpy() * 255).astype("uint8"))

    # Draw debugging information on the image
    gc = ImageDraw.Draw(out)
    gc.text((16, 16), (f"min={round(depth_min.item(), 4)}\n"
                       f"max={round(depth_max.item(), 4)}\n"
                       f"mean={round(mean_depth, 4)}\n"
                       f"std={round(std_depth, 4)}"), fill="gray")

    return out

def process_image(im, args, depth_model, side_model, return_tensor=False):
    with mx.inference_mode():
        # Preprocess the input image
        im_org, im = preprocess_image(im, args)

        # Infer the depth from the model
        depth = args.state["depth_utils"].batch_infer(
            depth_model, im, flip_aug=args.tta, low_vram=args.low_vram,
            int16=False, enable_amp=not args.disable_amp,
            output_device=args.state["device"],
            device=args.state["device"],
            edge_dilation=args.edge_dilation,
            resize_depth=False)

        if not args.debug_depth:
            # Apply divergence and post-process the image
            left_eye, right_eye = apply_divergence(depth, im_org.to(args.state["device"]), args, side_model)
            sbs = postprocess_image(left_eye, right_eye, args)

            if not return_tensor:
                # Convert to PIL image if requested
                sbs = Image.fromarray((sbs.numpy() * 255).astype("uint8").transpose(1, 2, 0))

            return sbs
        else:
            # Return a debug image if requested
            return debug_depth_image(depth, args, args.ema_normalize)

def process_images(files, output_dir, args, depth_model, side_model, title=None):
    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        # Skip existing output files
        remaining_files = []
        existing_files = []
        for fn in files:
            output_filename = os.path.join(
                output_dir,
                make_output_filename(os.path.basename(fn), args, video=False))
            if not os.path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"}
    )
    
    futures = []
    tqdm_fn = args.state.get("tqdm_fn", tqdm)
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title)
    
    stop_event = args.state.get("stop_event")
    suspend_event = args.state.get("suspend_event")
    
    with PoolExecutor(max_workers=4) as pool:
        for im, meta in loader:
            filename = meta["filename"]
            output_filename = os.path.join(
                output_dir,
                make_output_filename(filename, args, video=False)
            )
            if im is None:
                pbar.update(1)
                continue

            # Process the image
            output = process_image(im, args, depth_model, side_model)
            
            # Save the processed image using a thread pool
            f = pool.submit(save_image, output, output_filename)
            futures.append(f)
            pbar.update(1)
            
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []
        
        # Ensure all remaining futures are processed
        for f in futures:
            f.result()
    
    pbar.close()

def process_video_full(input_filename, output_path, args, depth_model, side_model):
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if side_model is not None:
        # TODO: sometimes ERROR RUNNING GUARDS forward error happen
        # side_model = compile_model(side_model, dynamic=True)
        pass

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(
            output_path,
            make_output_filename(os.path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    if args.resume and os.path.exists(output_filename):
        return

    if not args.yes and os.path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    make_parent_dir(output_filename)

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps

        frame_packing = "4" if args.tb or args.half_tb else "3"
        options = {"preset": args.preset, "crf": str(args.crf), "frame-packing": frame_packing}
        if args.tune:
            options["tune"] = ",".join(set(args.tune))
        if args.profile_level:
            options["level"] = args.profile_level
        if args.video_codec == "libx265":
            x265_params = ["log-level=warning", "high-tier=enabled"]
            if args.profile_level:
                x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")
            options["x265-params"] = ":".join(x265_params)

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            options=options if args.video_codec in {"libx264", "libx265"} else {},
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        )

    @mx.inference_mode()
    def test_callback(frame):
        frame = VU.to_frame(process_image(VU.to_tensor(frame), args, depth_model, side_model,
                                          return_tensor=True))
        if ema_normalize:
            args.state["ema"].clear()
        return frame

    if args.low_vram or args.debug_depth:
        @mx.inference_mode()
        def frame_callback(frame):
            if frame is None:
                return None
            return VU.to_frame(process_image(VU.to_tensor(frame), args, depth_model, side_model,
                                             return_tensor=True))

        VU.process_video(input_filename, output_filename,
                         config_callback=config_callback,
                         frame_callback=frame_callback,
                         test_callback=test_callback,
                         vf=args.vf,
                         stop_event=args.state["stop_event"],
                         suspend_event=args.state["suspend_event"],
                         tqdm_fn=args.state["tqdm_fn"],
                         title=os.path.basename(input_filename),
                         start_time=args.start_time,
                         end_time=args.end_time)
    else:
        minibatch_size = args.zoed_batch_size // 2 if args.tta else args.zoed_batch_size
        preprocess_lock = threading.Lock()
        depth_lock = threading.Lock()
        sbs_lock = threading.Lock()

        @mx.inference_mode()
        def _batch_callback(x):
            if args.max_output_height is not None or args.bg_session is not None:
                # TODO: batch preprocess_image
                with preprocess_lock:
                    xs = [preprocess_image(xx, args) for xx in x]
                    x = mx.stack([x for x_org, x in xs])
                    if args.bg_session is not None:
                        x_orgs = mx.stack([x_org for x_org, x in xs])
                    else:
                        x_orgs = x
            else:
                x_orgs = x

            with depth_lock:
                depths = args.state["depth_utils"].batch_infer(
                    depth_model, x, flip_aug=args.tta, low_vram=args.low_vram,
                    int16=False, enable_amp=not args.disable_amp,
                    output_device=args.state["device"],
                    device=args.state["device"],
                    edge_dilation=args.edge_dilation,
                    resize_depth=False)

            if args.method in {"forward", "forward_fill"}:
                # Lock all threads for forward warp
                with sbs_lock, preprocess_lock, depth_lock:
                    left_eyes, right_eyes = apply_divergence(depths, x_orgs, args, side_model, ema_normalize)
            else:
                with sbs_lock:
                    left_eyes, right_eyes = apply_divergence(depths, x_orgs, args, side_model, ema_normalize)

            return mx.stack([
                postprocess_image(left_eyes[i], right_eyes[i], args)
                for i in range(left_eyes.shape[0])])

        frame_callback = VU.FrameCallbackPool(
            _batch_callback,
            batch_size=minibatch_size,
            device=args.state["device"],
            max_workers=args.max_workers,
            max_batch_queue=args.max_workers + 1,
        )

        VU.process_video(input_filename, output_filename,
                         config_callback=config_callback,
                         frame_callback=frame_callback,
                         test_callback=test_callback,
                         vf=args.vf,
                         stop_event=args.state["stop_event"],
                         suspend_event=args.state["suspend_event"],
                         tqdm_fn=args.state["tqdm_fn"],
                         title=os.path.basename(input_filename),
                         start_time=args.start_time,
                         end_time=args.end_time)
        frame_callback.shutdown()

def process_video_keyframes(input_filename, output_path, args, depth_model, side_model):
    # Determine output filename and directory
    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(
            output_path,
            make_output_filename(os.path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    output_dir = os.path.join(os.path.dirname(output_filename), os.path.splitext(os.path.basename(output_filename))[0])
    if output_dir.endswith("_LRF"):
        output_dir = output_dir[:-4]
    os.makedirs(output_dir, exist_ok=True)

    with PoolExecutor(max_workers=4) as pool:
        futures = []

        def frame_callback(frame):
            # Process each keyframe
            output = process_image(frame.to_image(), args, depth_model, side_model)
            output_filename = os.path.join(
                output_dir,
                os.path.basename(output_dir) + "_" + str(frame.pts).zfill(8) + FULL_SBS_SUFFIX + ".png")
            f = pool.submit(save_image, output, output_filename)
            futures.append(f)

        # Process the video keyframes
        VU.process_video_keyframes(input_filename, frame_callback=frame_callback,
                                   min_interval_sec=args.keyframe_interval,
                                   stop_event=args.state["stop_event"],
                                   suspend_event=args.state["suspend_event"],
                                   title=os.path.basename(input_filename))

        # Ensure all futures are completed
        for f in futures:
            f.result()

def process_video(input_filename, output_path, args, depth_model, side_model):
    if args.keyframe:
        process_video_keyframes(input_filename, output_path, args, depth_model, side_model)
    else:
        process_video_full(input_filename, output_path, args, depth_model, side_model)


def export_images(args):
    # Determine if the input is a directory or a single image
    if os.path.isdir(args.input):
        files = ImageLoader.listdir(args.input)
        rgb_dir = os.path.normpath(os.path.abspath(args.input))
    else:
        assert is_image(args.input)
        files = [args.input]
        rgb_dir = os.path.normpath(os.path.abspath(os.path.dirname(args.input)))

    # Set configuration based on export type
    if args.export_disparity:
        mapper = "none"
        edge_dilation = args.edge_dilation
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        edge_dilation = 0
        skip_edge_dilation = False
        skip_mapper = False

    config = export_config.ExportConfig(
        type=export_config.IMAGE_TYPE,
        fps=1,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "ema_normalize": False,
            }
        }
    )
    config.rgb_dir = rgb_dir
    config.audio_file = None
    output_dir = args.output
    depth_dir = os.path.join(output_dir, config.depth_dir)
    config_file = os.path.join(output_dir, export_config.FILENAME)

    os.makedirs(depth_dir, exist_ok=True)
    depth_model = args.state["depth_model"]

    if args.resume:
        # Skip existing depth files
        remaining_files = []
        existing_files = []
        for fn in files:
            basename = os.path.splitext(os.path.basename(fn))[0] + ".png"
            depth_file = os.path.join(depth_dir, basename)
            if not os.path.exists(depth_file):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)

        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"}
    )
    futures = []
    tqdm_fn = args.state.get("tqdm_fn", tqdm)
    pbar = tqdm_fn(ncols=80, total=len(files), desc="Images")
    stop_event = args.state.get("stop_event")
    suspend_event = args.state.get("suspend_event")

    with PoolExecutor(max_workers=4) as pool, mx.inference_mode():
        for im, meta in loader:
            basename = os.path.splitext(os.path.basename(meta["filename"]))[0] + ".png"
            depth_file = os.path.join(depth_dir, basename)
            if im is None:
                pbar.update(1)
                continue
            
            # Preprocess the image
            im_org, im = preprocess_image(im, args)
            
            # Perform depth inference
            depth = args.state["depth_utils"].batch_infer(
                depth_model, im, flip_aug=args.tta, low_vram=args.low_vram,
                int16=False, enable_amp=not args.disable_amp,
                output_device=args.state["device"],
                device=args.state["device"],
                edge_dilation=edge_dilation,
                resize_depth=False
            )

            # Normalize and process depth
            depth = normalize_depth(depth)
            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            
            # Convert depth to uint16 numpy array and save as an image
            depth = convert_normalized_depth_to_uint16_numpy(depth[0])
            depth_image = Image.fromarray(depth)
            futures.append(pool.submit(save_image, depth_image, depth_file))
            pbar.update(1)

            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []

        for f in futures:
            f.result()

    pbar.close()
    config.save(config_file)

def get_resume_seq(depth_dir, rgb_dir):
    # Get sorted lists of depth and RGB files
    depth_files = sorted(os.listdir(depth_dir))
    rgb_files = sorted(os.listdir(rgb_dir))

    # Determine the last sequence based on the minimum of the last RGB and depth files
    if rgb_files and depth_files:
        last_seq = int(os.path.splitext(min(rgb_files[-1], depth_files[-1]))[0], 10)
    else:
        last_seq = -1  # No files found, return -1

    return last_seq

def export_video(args):
    basename = os.path.splitext(os.path.basename(args.input))[0]
    if args.export_disparity:
        mapper = "none"
        edge_dilation = args.edge_dilation
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        edge_dilation = 0
        skip_edge_dilation = False
        skip_mapper = False
    
    config = export_config.ExportConfig(
        type=export_config.VIDEO_TYPE,
        basename=basename,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "max_fps": args.max_fps,
                "ema_normalize": args.ema_normalize,
            }
        }
    )

    # Normalize the output directory path
    output_dir = os.path.join(args.output, basename.strip())
    rgb_dir = os.path.join(output_dir, config.rgb_dir)
    depth_dir = os.path.join(output_dir, config.depth_dir)
    audio_file = os.path.join(output_dir, config.audio_file)
    config_file = os.path.join(output_dir, export_config.FILENAME)

    # Handle overwrite protection
    if not args.resume and (not args.yes and os.path.exists(config_file)):
        y = input(f"File '{config_file}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    if args.resume:
        resume_seq = get_resume_seq(depth_dir, rgb_dir) - args.zoed_batch_size
    else:
        resume_seq = -1

    # Handle audio export
    if resume_seq > 0 and os.path.exists(audio_file):
        has_audio = True
    else:
        has_audio = VU.export_audio(args.input, audio_file,
                                    start_time=args.start_time, end_time=args.end_time,
                                    title="Audio",
                                    stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
                                    tqdm_fn=args.state["tqdm_fn"])
    if not has_audio:
        config.audio_file = None

    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
        return

    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        args.state["ema"].clear()

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        config.fps = fps  # Update fps

        def state_update_callback(c):
            config.source_color_range = c.state["source_color_range"]
            config.output_colorspace = c.state["output_colorspace"]

        video_output_config = VU.VideoOutputConfig(fps=fps, pix_fmt=args.pix_fmt, colorspace=args.colorspace)
        video_output_config.state_updated = state_update_callback

        return video_output_config

    minibatch_size = args.zoed_batch_size // 2 if args.tta else args.zoed_batch_size
    preprocess_lock = threading.Lock()
    depth_lock = threading.Lock()
    depth_model = args.state["depth_model"]

    @mx.inference_mode()
    def _batch_callback(x, pts):
        if args.max_output_height is not None or args.bg_session is not None:
            with preprocess_lock:
                xs = [preprocess_image(xx, args) for xx in x]
                x = mx.stack([x for x_org, x in xs])
                if args.bg_session is not None:
                    x_orgs = mx.stack([x_org for x_org, x in xs])
                else:
                    x_orgs = x
        else:
            x_orgs = x

        with depth_lock:
            depths = args.state["depth_utils"].batch_infer(
                depth_model, x,
                int16=False,
                flip_aug=args.tta, low_vram=args.low_vram,
                enable_amp=not args.disable_amp,
                output_device=args.state["device"],
                device=args.state["device"],
                edge_dilation=edge_dilation,
                resize_depth=False)

            for i in range(depths.shape[0]):
                depth_min, depth_max = mx.min(depths[i]), mx.max(depths[i])
                if ema_normalize:
                    depth_min, depth_max = args.state["ema"].update(depth_min, depth_max)
                depth = normalize_depth(depths[i], depth_min=depth_min, depth_max=depth_max)
                if args.export_disparity:
                    depth = get_mapper(args.mapper)(depth)
                depths[i] = depth

        depths = depths.detach().cpu()
        x_orgs = x_orgs.detach().cpu()

        for x, depth, seq in zip(x_orgs, depths, pts):
            seq = str(seq).zfill(8)
            depth = convert_normalized_depth_to_uint16_numpy(depth[0])
            depth_image = Image.fromarray(depth)
            depth_image.save(os.path.join(depth_dir, f"{seq}.png"))
            rgb = TF.to_pil_image(x)
            rgb.save(os.path.join(rgb_dir, f"{seq}.png"))

    frame_callback = VU.FrameCallbackPool(
        _batch_callback,
        batch_size=minibatch_size,
        device=args.state["device"],
        max_workers=args.max_workers,
        max_batch_queue=args.max_workers + 1,
        require_pts=True,
        skip_pts=resume_seq
    )

    VU.hook_frame(args.input,
                  config_callback=config_callback,
                  frame_callback=frame_callback,
                  vf=args.vf,
                  stop_event=args.state["stop_event"],
                  suspend_event=args.state["suspend_event"],
                  tqdm_fn=args.state["tqdm_fn"],
                  title=os.path.basename(args.input),
                  start_time=args.start_time,
                  end_time=args.end_time)

    frame_callback.shutdown()
    config.save(config_file)

def to_float32_grayscale_depth(depth):
    if depth.dtype != mx.float32:
        # Handle 16-bit depth images
        depth = mx.clip(depth.astype(mx.float32) / 0xFFFF, 0, 1)

    if depth.shape[0] != 1:
        # Handle non-single-channel images by averaging across channels
        # TODO: Implement color depth support if needed
        depth = mx.mean(depth, axis=0, keepdims=True)

    # Invert the depth values
    depth = 1.0 - depth
    return depth

def process_config_video(config, args, side_model):
    base_dir = os.path.dirname(args.input)
    rgb_dir, depth_dir, audio_file = config.resolve_paths(base_dir)

    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        basename = config.basename or os.path.basename(base_dir)
        output_filename = os.path.join(
            args.output,
            make_output_filename(basename, args, video=True))
    else:
        output_filename = args.output
    make_parent_dir(output_filename)

    if args.resume and os.path.exists(output_filename):
        return
    if not args.yes and os.path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match: rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "any"})

    sbs_lock = threading.Lock()

    @mx.inference_mode()
    def batch_callback(x, depths):
        if not config.skip_edge_dilation and args.edge_dilation > 0:
            # Apply edge dilation
            depths = -dilate_edge(-depths, args.edge_dilation)
        with sbs_lock:
            left_eyes, right_eyes = apply_divergence(depths, x, args, side_model)
        return mx.stack([
            postprocess_image(left_eyes[i], right_eyes[i], args)
            for i in range(left_eyes.shape[0])])

    def test_output_size(rgb_file, depth_file):
        rgb = load_image_simple(rgb_file, color="rgb")[0]
        depth = load_image_simple(depth_file, color="any")[0]
        rgb = mx.array(rgb).astype(mx.float32) / 255.0  # Convert to tensor and normalize
        depth = to_float32_grayscale_depth(mx.array(depth))
        frame = batch_callback(rgb[None, ...].to(args.state["device"]),
                               depth[None, ...].to(args.state["device"]))
        return frame.shape[2:]

    minibatch_size = args.zoed_batch_size // 2 if args.tta else args.zoed_batch_size

    def generator():
        rgb_batch = []
        depth_batch = []
        for rgb, depth in zip(rgb_loader, depth_loader):
            rgb = mx.array(rgb[0]).astype(mx.float32) / 255.0
            depth = to_float32_grayscale_depth(mx.array(depth[0]))
            rgb_batch.append(rgb)
            depth_batch.append(depth)
            if len(rgb_batch) == minibatch_size:
                frames = batch_callback(mx.stack(rgb_batch).to(args.state["device"]),
                                        mx.stack(depth_batch).to(args.state["device"]))
                rgb_batch.clear()
                depth_batch.clear()
                yield [VU.to_frame(frame) for frame in frames]

        if rgb_batch:
            frames = batch_callback(mx.stack(rgb_batch).to(args.state["device"]),
                                    mx.stack(depth_batch).to(args.state["device"]))
            rgb_batch.clear()
            depth_batch.clear()
            yield [VU.to_frame(frame) for frame in frames]

    output_height, output_width = test_output_size(rgb_files[0], depth_files[0])
    frame_packing = "4" if args.tb or args.half_tb else "3"
    encoder_options = {"preset": args.preset, "crf": str(args.crf), "frame-packing": frame_packing}
    if args.tune:
        encoder_options.update({"tune": ",".join(list(set(args.tune)))})
    if args.profile_level:
        encoder_options["level"] = args.profile_level
    if args.video_codec == "libx265":
        x265_params = ["log-level=warning", "high-tier=enabled"]
        if args.profile_level:
            x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")
        encoder_options["x265-params"] = ":".join(x265_params)

    video_config = VU.VideoOutputConfig(
        fps=config.fps,  # use config.fps, ignore args.max_fps
        container_format=args.video_format,
        video_codec=args.video_codec,
        pix_fmt=args.pix_fmt,
        colorspace=args.colorspace,
        options=encoder_options if args.video_codec in {"libx264", "libx265"} else {},
        container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        output_width=output_width,
        output_height=output_height
    )
    video_config.state["source_color_range"] = config.source_color_range
    video_config.state["output_colorspace"] = config.output_colorspace

    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            # Force use of "none" mapper
            args.mapper = "none"
        else:
            if config.mapper is not None:
                # Use specified mapper from config
                args.mapper = config.mapper

        VU.generate_video(
            output_filename,
            generator,
            config=video_config,
            audio_file=audio_file,
            title=os.path.basename(base_dir),
            total_frames=len(rgb_files),
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
        )
    finally:
        args.mapper = original_mapper

def process_config_images(config, args, side_model):
    base_dir = os.path.dirname(args.input)
    rgb_dir, depth_dir, _ = config.resolve_paths(base_dir)

    def fix_rgb_depth_pair(files1, files2):
        # Align the RGB and depth file lists based on matching filenames
        db1 = {os.path.basename(fn): fn for fn in files1}
        db2 = {os.path.basename(fn): fn for fn in files2}
        files2 = [fn for key, fn in db2.items() if key in db1]
        files1 = [fn for key, fn in db1.items() if key in db2]
        return files1, files2

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)

    if args.resume:
        # Skip existing output files
        remaining_files = []
        existing_files = []
        for fn in rgb_files:
            output_filename = os.path.join(
                output_dir,
                make_output_filename(os.path.basename(fn), args, video=False))
            if not os.path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        rgb_files = remaining_files

    rgb_files, depth_files = fix_rgb_depth_pair(rgb_files, depth_files)

    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match: rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "any"})

    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            args.mapper = "none"
        else:
            if config.mapper is not None:
                args.mapper = config.mapper

        with PoolExecutor(max_workers=4) as pool:
            tqdm_fn = args.state.get("tqdm_fn", tqdm)
            pbar = tqdm_fn(ncols=80, total=len(rgb_files), desc="Images")
            stop_event = args.state.get("stop_event")
            suspend_event = args.state.get("suspend_event")
            futures = []

            for (rgb, rgb_meta), (depth, depth_meta) in zip(rgb_loader, depth_loader):
                rgb_filename = os.path.splitext(os.path.basename(rgb_meta["filename"]))[0]
                depth_filename = os.path.splitext(os.path.basename(depth_meta["filename"]))[0]
                if rgb_filename != depth_filename:
                    raise ValueError(f"No match: {rgb_filename} and {depth_filename}")

                # Convert images to MLX tensors
                rgb = mx.array(rgb).astype(mx.float32) / 255.0
                depth = to_float32_grayscale_depth(mx.array(depth))

                # Apply edge dilation if necessary
                if not config.skip_edge_dilation and args.edge_dilation > 0:
                    depth = -dilate_edge(-depth[None, ...], args.edge_dilation).squeeze(0)

                # Apply divergence and post-process the images
                left_eye, right_eye = apply_divergence(
                    depth.to(args.state["device"]),
                    rgb.to(args.state["device"]),
                    args, side_model)
                sbs = postprocess_image(left_eye, right_eye, args)
                sbs = Image.fromarray((sbs.numpy() * 255).astype("uint8").transpose(1, 2, 0))

                # Save the processed image
                output_filename = os.path.join(
                    output_dir,
                    make_output_filename(rgb_filename, args, video=False))
                f = pool.submit(save_image, sbs, output_filename)
                futures.append(f)
                pbar.update(1)

                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
                if len(futures) > IMAGE_IO_QUEUE_MAX:
                    for f in futures:
                        f.result()
                    futures = []

            for f in futures:
                f.result()

            pbar.close()
    finally:
        args.mapper = original_mapper

def create_parser(required_true=True):
    class Range:
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

        def __repr__(self):
            return f"{self.start} <= value <= {self.end}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Default GPU based on availability
    default_gpu = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1

    parser.add_argument("--input", "-i", type=str, required=required_true, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, required=required_true, help="Output file or directory")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu], help="GPU device ID. -1 for CPU")
    
    parser.add_argument("--method", type=str, default="row_flow",
                        choices=["grid_sample", "backward", "forward", "forward_fill", "row_flow",
                                 "row_flow_sym", "row_flow_v3", "row_flow_v3_sym", "row_flow_v2"],
                        help="Left-right divergence method")
    parser.add_argument("--divergence", "-d", type=float, default=2.0, help="Strength of 3D effect. 0-2 is reasonable")
    parser.add_argument("--convergence", "-c", type=float, default=0.5,
                        help="(Normalized) distance of convergence plane (screen position). 0-1 is reasonable")
    parser.add_argument("--update", action="store_true", help="Force update MiDaS models from Torch Hub")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process all subdirectories")
    parser.add_argument("--resume", action="store_true", help="Skip processing when the output file already exists")
    parser.add_argument("--batch-size", type=int, default=16, choices=[Range(1, 256)],
                        help="Batch size for RowFlow model. !!DEPRECATED!!")
    parser.add_argument("--zoed-batch-size", type=int, default=2, choices=[Range(1, 64)],
                        help="Batch size for ZoeDepth model. Ignored when --low-vram")
    parser.add_argument("--max-fps", type=float, default=30, help="Max framerate for video")
    parser.add_argument("--profile-level", type=str, help="H264 profile level")
    parser.add_argument("--crf", type=int, default=20, help="Constant quality value for video. Smaller is higher quality")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
                                 "veryslow", "placebo"], help="Encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        choices=["film", "animation", "grain", "stillimage", "psnr", "fastdecode", "zerolatency"],
                        help="Encoder tunings option for video")
    parser.add_argument("--yes", "-y", action="store_true", default=False, help="Overwrite output files")
    parser.add_argument("--pad", type=float, help="Pad size = int(size * pad)")
    parser.add_argument("--depth-model", type=str, default="ZoeD_N",
                        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK", "Any_S", "Any_B", "Any_L", "ZoeD_Any_N", "ZoeD_Any_K",
                                 "Any_V2_S", "Any_V2_B", "Any_V2_L", "Any_V2_N", "Any_V2_K", "Any_V2_N_S",
                                 "Any_V2_N_B", "Any_V2_N_L", "Any_V2_K_S", "Any_V2_K_B", "Any_V2_K_L"],
                        help="Depth model name")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background depth, not recommended for video")
    parser.add_argument("--bg-model", type=str, default="u2net_human_seg", help="Rembg model type")
    parser.add_argument("--rotate-left", action="store_true", help="Rotate 90 degrees to the left (counterclockwise)")
    parser.add_argument("--rotate-right", action="store_true", help="Rotate 90 degrees to the right (clockwise)")
    parser.add_argument("--low-vram", action="store_true", help="Disable batch processing for low memory GPU")
    parser.add_argument("--keyframe", action="store_true", help="Process only keyframes as images")
    parser.add_argument("--keyframe-interval", type=float, default=4.0, help="Keyframe minimum interval (sec)")
    parser.add_argument("--vf", type=str, default="", help="Video filter options for ffmpeg.")
    parser.add_argument("--debug-depth", action="store_true",
                        help="Debug output normalized depthmap, info and preprocessed depth")
    parser.add_argument("--export", action="store_true", help="Export depth, frame, audio")
    parser.add_argument("--export-disparity", action="store_true",
                        help="Export disparity instead of depth. Applies --mapper and --foreground-scale.")
    parser.add_argument("--mapper", type=str,
                        choices=["auto", "pow2", "softplus", "softplus2", "div_6", "div_4", "div_2", "div_1", "none",
                                 "mul_1", "mul_2", "mul_3", "inv_mul_1", "inv_mul_2", "inv_mul_3"],
                        help="(Re-)mapper function for depth. Deprecated; use --foreground-scale instead.")
    parser.add_argument("--foreground-scale", type=int, choices=[-3, -2, -1, 0, 1, 2, 3], default=0,
                        help="Foreground scaling level. 0 is disabled")
    parser.add_argument("--vr180", action="store_true", help="Output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true", help="Output in Half SBS")
    parser.add_argument("--tb", action="store_true", help="Output in Full TopBottom")
    parser.add_argument("--half-tb", action="store_true", help="Output in Half TopBottom")

    parser.add_argument("--anaglyph", type=str, nargs="?", default=None, const="dubois",
                        choices=["color", "gray", "half-color", "wimmer", "wimmer2", "dubois", "dubois2"],
                        help="Output in anaglyph 3D")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p", choices=["yuv420p", "yuv444p", "rgb24", "gbrp"],
                        help="Pixel format (video only)")
    parser.add_argument("--tta", action="store_true", help="Use flip augmentation on depth model")
    parser.add_argument("--disable-amp", action="store_true", help="Disable AMP for specific cases")
    parser.add_argument("--max-output-width", type=int, help="Limit output width for cardboard players")
    parser.add_argument("--max-output-height", type=int, help="Limit output height for cardboard players")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="Keep aspect ratio when resizing")
    parser.add_argument("--start-time", type=str, help="Set start time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--end-time", type=str, help="Set end time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--zoed-height", type=int, help="Input resolution (small side) for depth model")
    parser.add_argument("--stereo-width", type=int, help="Input width for row_flow_v3/row_flow_v2 model")
    parser.add_argument("--ipd-offset", type=float, default=0, help="IPD Offset (width scale %%). 0-10 is reasonable")
    parser.add_argument("--ema-normalize", action="store_true", help="Use min/max moving average to normalize video depth")
    parser.add_argument("--ema-decay", type=float, default=0.75, help="Parameter for EMA normalize (0-1)")
    parser.add_argument("--edge-dilation", type=int, nargs="?", default=None, const=2, help="Loop count of edge dilation")
    parser.add_argument("--max-workers", type=int, default=0, choices=[0, 1, 2, 3, 4, 8, 16],
                        help="Max inference worker threads for video processing. 0 disables")
        parser.add_argument("--video-format", "-vf", type=str, default="mp4",
                        choices=["mp4", "mkv", "avi"], help="Video container format")
    parser.add_argument("--video-codec", "-vc", type=str, default=None, help="Video codec")

    parser.add_argument("--metadata", type=str, nargs="?", default=None, const="filename", choices=["filename"],
                        help="Add metadata")
    parser.add_argument("--find-param", type=str, nargs="+",
                        choices=["divergence", "convergence", "foreground-scale", "ipd-offset"],
                        help="Output results for various parameter combinations")

    # TODO: Change the default value from "unspecified" to "auto"
    parser.add_argument("--colorspace", type=str, default="unspecified",
                        choices=["unspecified", "auto", "bt709", "bt709-pc", "bt709-tv", "bt601", "bt601-pc", "bt601-tv"],
                        help="Video colorspace")

    return parser

class EMAMinMax():
    def __init__(self, alpha=0.75):
        self.min = None
        self.max = None
        self.alpha = alpha

    def update(self, min_value, max_value):
        if self.min is None:
            self.min = float(min_value)
            self.max = float(max_value)
        else:
            self.min = self.alpha * self.min + (1. - self.alpha) * float(min_value)
            self.max = self.alpha * self.max + (1. - self.alpha) * float(max_value)

        # print(round(float(min_value), 3), round(float(max_value), 3), round(self.min, 3), round(self.max, 3))

        return self.min, self.max

    def clear(self):
        self.min = self.max = None

def set_state_args(args, stop_event=None, tqdm_fn=None, depth_model=None, suspend_event=None):
    from . import zoedepth_model as ZU
    from . import depth_anything_model as DU
    
    # Select the appropriate depth utilities based on the model
    if args.depth_model in ZU.MODEL_FILES:
        depth_utils = ZU
    elif args.depth_model in DU.MODEL_FILES:
        depth_utils = DU
    else:
        raise ValueError(f"Unknown depth model: {args.depth_model}")

    # Enable export if exporting disparity
    if args.export_disparity:
        args.export = True

    # Adjust video format based on output file extension
    if is_video(args.output):
        ext = path.splitext(args.output)[-1]
        if ext == ".mp4":
            args.video_format = "mp4"
        elif ext == ".mkv":
            args.video_format = "mkv"
        elif ext == ".avi":
            args.video_format = "avi"

    # Set the video extension and codec
    args.video_extension = f".{args.video_format}"
    if args.video_codec is None:
        args.video_codec = VU.get_default_video_codec(args.video_format)

    # Handle auto or unspecified profile levels
    if not args.profile_level or args.profile_level == "auto":
        args.profile_level = None

    # Initialize the state dictionary
    args.state = {
        "stop_event": stop_event,
        "suspend_event": suspend_event,
        "tqdm_fn": tqdm_fn,
        "depth_model": depth_model,
        "ema": EMAMinMax(alpha=args.ema_decay),
        "device": create_device(args.gpu),
        "depth_utils": depth_utils,
    }
    
    return args

def export_main(args):
    # Handle unsupported cases
    if args.recursive:
        raise NotImplementedError("`--recursive --export` is not supported")
    if is_text(args.input):
        raise NotImplementedError("--export with text format input is not supported")

    # Route to the appropriate export function based on input type
    if path.isdir(args.input) or is_image(args.input):
        export_images(args)
    elif is_video(args.input):
        export_video(args)
    else:
        raise ValueError("Unrecognized file type")

def is_yaml(filename):
    return path.splitext(filename)[-1].lower() in {".yaml", ".yml"}


def iw3_main(args):
    # Validate conflicting arguments
    assert not (args.rotate_left and args.rotate_right)
    assert not (args.half_sbs and args.vr180)
    assert not (args.half_sbs and args.anaglyph)
    assert not (args.vr180 and args.anaglyph)

    # Force update of depth models if requested
    if args.update:
        args.state["depth_utils"].force_update()

    # Ensure input and output are different
    if os.path.normpath(args.input) == os.path.normpath(args.output):
        raise ValueError("Input and output must be different files")

    # Handle YAML input with export flag
    if args.export and is_yaml(args.input):
        raise ValueError("YAML file input does not support --export")

    # Validate tune options for libx265 codec
    if args.tune and args.video_codec == "libx265":
        if len(args.tune) != 1:
            raise ValueError("libx265 does not support multiple --tune options.\n"
                             f"tune={','.join(args.tune)}")
        if args.tune[0] in {"film", "stillimage"}:
            raise ValueError(f"libx265 does not support --tune {args.tune[0]}\n"
                             "Available options: grain, animation, psnr, zerolatency, fastdecode")

    # Set up background removal if requested
    if args.remove_bg:
        global rembg
        import rembg
        args.bg_session = rembg.new_session(model_name=args.bg_model)
    else:
        args.bg_session = None

    # Determine edge dilation
    if args.edge_dilation is None:
        if args.state["depth_utils"].get_name() == "DepthAnything":
            args.edge_dilation = 2
        else:
            args.edge_dilation = 0

    # Load or set depth model
    if not is_yaml(args.input):
        if args.state["depth_model"] is not None:
            depth_model = args.state["depth_model"]
        else:
            depth_model = args.state["depth_utils"].load_model(model_type=args.depth_model, gpu=args.gpu,
                                                               height=args.zoed_height)
            args.state["depth_model"] = depth_model

        is_metric = (args.state["depth_utils"].get_name() == "ZoeDepth" or
                     (args.state["depth_utils"].get_name() == "DepthAnything" and args.state["depth_model"].metric_depth))
        args.mapper = resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                          metric_depth=is_metric)
    else:
        depth_model = None
        args.mapper = "none"

    # Handle export
    if args.export:
        export_main(args)
        return args

    # Load the side model based on the method
    with TorchHubDir(HUB_MODEL_DIR):
        if args.method in {"row_flow_v3", "row_flow"}:
            side_model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = False
            side_model.delta_output = True
        elif args.method in {"row_flow_v3_sym", "row_flow_sym"}:
            side_model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.symmetric = True
            side_model.delta_output = True
        elif args.method == "row_flow_v2":
            side_model = load_model(ROW_FLOW_V2_URL, weights_only=True, device_ids=[args.gpu[0]])[0].eval()
            side_model.delta_output = True
        else:
            side_model = None

    # Handle parameter search
    if args.find_param:
        assert is_image(args.input) and (os.path.isdir(args.output) or not os.path.exists(args.output))
        find_param(args, depth_model, side_model)
        return args

    # Process input based on its type
    if os.path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            image_files = ImageLoader.listdir(args.input)
            process_images(image_files, args.output, args, depth_model, side_model, title="Images")
            for video_file in VU.list_videos(args.input):
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return args
                process_video(video_file, args.output, args, depth_model, side_model)
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = os.path.normpath(os.path.join(args.output, os.path.relpath(input_dir, start=args.input)))
                image_files = ImageLoader.listdir(input_dir)
                if image_files:
                    process_images(image_files, output_dir, args, depth_model, side_model,
                                   title=os.path.relpath(input_dir, args.input))
                for video_file in VU.list_videos(input_dir):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    process_video(video_file, output_dir, args, depth_model, side_model)

    elif is_yaml(args.input):
        config = export_config.ExportConfig.load(args.input)
        if config.type == export_config.VIDEO_TYPE:
            process_config_video(config, args, side_model)
        if config.type == export_config.IMAGE_TYPE:
            process_config_images(config, args, side_model)
    elif is_text(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        files = []
        with open(args.input, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                files.append(line.strip())
        image_files = [f for f in files if is_image(f)]
        process_images(image_files, args.output, args, depth_model, side_model, title="Images")
        video_files = [f for f in files if is_video(f)]
        for video_file in video_files:
            if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                return args
            process_video(video_file, args.output, args, depth_model, side_model)
    elif is_video(args.input):
        process_video(args.input, args.output, args, depth_model, side_model)
    elif is_image(args.input):
        if is_output_dir(args.output):
            os.makedirs(args.output, exist_ok=True)
            output_filename = os.path.join(
                args.output,
                make_output_filename(args.input, args, video=False))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb")
        output = process_image(im, args, depth_model, side_model)
        make_parent_dir(output_filename)
        output.save(output_filename)
    else:
        raise ValueError("Unrecognized file type")

    return args

def find_param(args, depth_model, side_model):
    im, _ = load_image_simple(args.input, color="rgb")
    args.metadata = "filename"
    os.makedirs(args.output, exist_ok=True)

    # Define parameter ranges based on the method and args.find_param
    if args.method == "forward_fill":
        divergence_cond = range(1, 11) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(-2, 2.25, 0.25) if "convergence" in args.find_param else [args.convergence]
    else:
        divergence_cond = range(1, 5) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(0, 1.25, 0.25) if "convergence" in args.find_param else [args.convergence]

    foreground_scale_cond = range(0, 4) if "foreground-scale" in args.find_param else [args.foreground_scale]
    ipd_offset_cond = range(0, 6) if "ipd-offset" in args.find_param else [args.ipd_offset]

    # Loop over all combinations of parameters
    for divergence in divergence_cond:
        for convergence in convergence_cond:
            for foreground_scale in foreground_scale_cond:
                for ipd_offset in ipd_offset_cond:
                    # Update args with current parameter values
                    args.divergence = float(divergence)
                    args.convergence = float(convergence)
                    args.foreground_scale = foreground_scale
                    args.ipd_offset = ipd_offset

                    # Generate output filename based on current parameters
                    output_filename = os.path.join(
                        args.output,
                        make_output_filename("param.png", args, video=False))

                    print(f"Processing with divergence={divergence}, convergence={convergence}, "
                          f"foreground_scale={foreground_scale}, ipd_offset={ipd_offset}")
                    
                    # Process the image with the current parameters
                    output = process_image(im, args, depth_model, side_model)
                    
                    # Save the output image
                    output.save(output_filename)
