from sutra.logger import message

import time

import numpy as np
from typing import Callable, List, Tuple , Any , Literal
import math



def vram_input(batch_size: int, h: int, w: int, channels: int = 1,
               dtype=np.float32) -> float:
    """Return required VRAM in MiB for the input tensor."""
    bytes_per_elem = np.dtype(dtype).itemsize          # 4 for float32
    total_bytes = batch_size * h * w * channels * bytes_per_elem
    return total_bytes / (1024 ** 2)                    # MiB

# # Example
# print(vram_input(1024, 64, 64, channels=1))   # 16.0 MiB
# print(vram_input(1024, 64, 64, channels=3))   # 48.0 MiB

def count_patches(image_shape, chunk_shape, overlap):
    message("Computing number of Patches", 'i')
    H, W = image_shape[:2]
    step_h, step_w = chunk_shape
    # ovlp = int(chunk_shape[0] * window_overlap_frac)   # symmetric overlap
    # overlap = (ovlp, ovlp)
    ov_h, ov_w     = overlap
    stride_h = step_h - ov_h
    stride_w = step_w - ov_w

    total_rows = (H - step_h) // stride_h + 1
    # message(total_rows , 'i')
    n = 0
    for row_idx in range(int(total_rows)):
        half_shift = stride_h // 2 if row_idx % 2 else 0
        c = 0
        while True:
            c0 = c + half_shift
            if c0 + step_w > W:
                break
            n += 1
            c += stride_w
    return n


def benchmark_preproc(preproc_fns, sample_patches, n_warm=1, n_rep=1, n_jobs=1):
    message("Estimating Preprocessing Time", 'i')

    """
    Returns the average time (seconds) to run all `preproc_fns`
    on ONE patch (or on a small batch if you prefer).
    """
    # ---- warm‑up -------------------------------------------------
    for _ in range(n_warm):
        for patch in sample_patches:
            p = patch.copy()
            for fn in preproc_fns:
                p = fn(p)

    # ---- timed runs ---------------------------------------------
    timings = []
    for _ in range(n_rep):
        start = time.perf_counter()
        for patch in sample_patches:
            p = patch.copy()
            for fn in preproc_fns:
                p = fn(p)
        timings.append(time.perf_counter() - start)

    avg_per_patch = sum(timings) / (len(sample_patches) * n_rep)
    return avg_per_patch

import tensorflow as tf

def benchmark_inference(model, batch_shape, batch_size,
                        n_warm=2, n_rep=3, verbose=0):
    message("Estimating Model run Time", 'i')
    
    dummy = np.random.rand(*batch_shape).astype('float32')
    # warm‑up
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 64, 64, 1], dtype=tf.float32)])
    def wrapped_predict(x):
        return model(x, training=False)
    _ = wrapped_predict(tf.random.uniform((1, 64, 64, 1)))
    # message("Estimating Modfddfdfdel run Time", 'i')

    # timed runs
    timings = []
    for _ in range(n_rep):
        start = time.perf_counter()
        _ = wrapped_predict(dummy)
        timings.append(time.perf_counter() - start)

    return sum(timings) / len(timings)   # seconds per batch




def estimate_prediction_computation_time(
    image: np.ndarray,
    model: Any,
    chunk_shape: Tuple[int, int],
    overlap: Tuple[int, int],
    global_norm : List[Callable] | None = None,
    preproc_fns: List[Callable] | None = None,
    batch_size: int | None = None,
    n_jobs: int = 1,
    *,
    initializer: Callable | None = None,
    initargs: Tuple[Any, ...] = (),
) -> np.ndarray:
    sample_raw = []
    H, W = image.shape[:2]
    step_h, step_w = chunk_shape
    ov_h, ov_w = overlap
    stride_h = step_h - ov_h
    stride_w = step_w - ov_w

    total_rows = (H - step_h) // stride_h + 1
    for row_idx in range(total_rows):
        r0 = row_idx * stride_h
        half_shift = stride_h // 2 if row_idx % 2 else 0
        c = 0
        while True:
            c0 = c + half_shift
            if c0 + step_w > W:
                break
            patch = image[r0:r0+step_h, c0:c0+step_w]
            sample_raw.append(patch)
            if len(sample_raw) >= 10:          # stop after ~100 patches
                break
            c += stride_w
        if len(sample_raw) >= 10:
            break
    # after you have `raw_patches[:100]` (or the `sample_raw` built above)
    if preproc_fns:
        t_preproc_per_patch = benchmark_preproc(preproc_fns, sample_raw)
    else:
        t_preproc_per_patch = 0.0
    # print(t_preproc_per_patch)

    # inference benchmark (use the same batch_size you will run later)
    channel = image.shape[2] if image.ndim == 3 else 1
    batch_shape = (batch_size, *chunk_shape, channel)
    t_per_batch = benchmark_inference(model, batch_shape, batch_size)

    # analytic patch count & total time
    total_patches = count_patches(image.shape, chunk_shape, overlap)
    n_batches = math.ceil(total_patches / batch_size)

    t_preproc_total = t_preproc_per_patch * total_patches
    t_inference_total = t_per_batch * n_batches

    deets = {}

    vram_req = vram_input(batch_size=batch_size, h=chunk_shape[0], w=chunk_shape[1], channels=1)

    deets.update({
        "Total patches": total_patches,
        # "Avg pre‑proc / patch (s)": round(t_preproc_per_patch, 6),
        # "Pre‑proc total (s)": round(t_preproc_total, 2),
        # "Avg inference / batch (s)": round(t_per_batch, 3),
        # "Inference total (s)": round(t_inference_total, 2),
        "Estimated total (s)": round(t_preproc_total + t_inference_total, 2),
        "Estimated total (min)": round((t_preproc_total + t_inference_total)/60, 2),
        # "VRAM (MB)" : vram_req,
    })
    return deets