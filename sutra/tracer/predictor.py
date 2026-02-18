import os
import numpy as np
from typing import Callable, List, Tuple, Any, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import streamlit as st


from sutra.logger import message
# ----------------------------------------------------------------------
#  INTERNAL worker – now applies preproc_fns *before* model.predict
# ----------------------------------------------------------------------
def _worker_process_rows(
    rows_range: Tuple[int, int],
    image: np.ndarray,
    model: Any,
    chunk_shape: Tuple[int, int],
    overlap: Tuple[int, int],
    preproc_fns: List[Callable] | None,
    batch_size: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the partial sum‑canvas and count‑canvas for the rows in
    ``rows_range`` (inclusive start, exclusive end).  All heavy work
    (model prediction) is performed inside the worker, therefore the model
    does not have to be picklable – you can load it via an initializer
    (see the note at the bottom of the file).
    """
    r_start, r_end = rows_range
    step_h, step_w = chunk_shape
    ov_h, ov_w = overlap
    stride_h = step_h - ov_h
    stride_w = step_w - ov_w
    half_stride_h = stride_h // 2
    half_stride_w = stride_w // 2

    # --------------------------------------------------------------
    # 1️⃣  Gather *raw* patches that belong to this worker
    # --------------------------------------------------------------
    raw_patches = []          # list of 2‑D (or 3‑D) arrays, one per patch
    coords      = []          # (r0, c0) for each patch – needed later

    for row_idx in range(r_start, r_end):
        r0 = row_idx * stride_h
        half_shift = half_stride_h if row_idx % 2 else 0
        c_start = 0
        while c_start < image.shape[1]:
            c0 = c_start + half_shift
            if c0 + step_w > image.shape[1]:
                break
            patch = image[r0 : r0 + step_h, c0 : c0 + step_w]
            raw_patches.append(patch)
            coords.append((r0, c0))
            c_start += stride_w

    # --------------------------------------------------------------
    # 2️⃣  OPTIONAL *per‑patch* preprocessing – **before** batching
    # --------------------------------------------------------------
    if preproc_fns:
        # Apply the list of functions sequentially to every patch.
        # Each function receives a **single patch** and must return a patch
        # of the same shape (or a shape that the model expects).
        processed = []
        for patch in raw_patches:
            for fn in preproc_fns:
                patch = fn(patch)
            processed.append(patch)
        raw_patches = processed                     # replace with processed ones

    # --------------------------------------------------------------
    # 3️⃣  Stack the (now pre‑processed) patches into a batch
    # --------------------------------------------------------------
    if len(raw_patches) == 0:                       # nothing to do in this slice
        sum_canvas = np.zeros_like(image, dtype=np.float64)
        cnt_canvas = np.zeros_like(image, dtype=np.float64)
        return sum_canvas, cnt_canvas

    batch = np.stack(raw_patches, axis=0).astype('float32')
    # Keras Conv2D‑style models usually expect a channel axis.
    # If the model expects a channel dimension and the image is 2‑D,
    # we add a singleton channel.
    if batch.ndim == 3:          # (N, H, W) → (N, H, W, 1)
        batch = batch[..., np.newaxis]

    # --------------------------------------------------------------
    # 4️⃣  Batched inference (same sub‑batch logic as before)
    # --------------------------------------------------------------
    if batch_size is None:
        preds = model.predict(batch, verbose=0).squeeze()
    else:
        chunks = []
        for start in range(0, batch.shape[0], batch_size):
            sub = batch[start:start + batch_size]
            p   = model.predict(sub, verbose=0).squeeze()
            chunks.append(p)
        preds = np.concatenate(chunks, axis=0)

    # --------------------------------------------------------------
    # 5️⃣  Write predictions into the per‑worker canvases
    # --------------------------------------------------------------
    sum_canvas = np.zeros_like(image, dtype=np.float64)
    cnt_canvas = np.zeros_like(image, dtype=np.float64)

    # If the model returns a channel dimension (N, H, W, C) we drop it.
    if preds.ndim == 4:                 # (N, h, w, C)
        preds = preds.squeeze(axis=-1)

    for (r0, c0), pred in zip(coords, preds):
        sum_canvas[r0:r0 + step_h, c0:c0 + step_w] += pred
        cnt_canvas[r0:r0 + step_h, c0:c0 + step_w] += 1.0

    return sum_canvas, cnt_canvas


# ----------------------------------------------------------------------
#  PUBLIC API – unchanged signature, only the internal logic moved
# ----------------------------------------------------------------------
def process_image_staggered_mean(
    image: np.ndarray,
    model: Any,
    chunk_shape: Tuple[int, int],
    overlap: Tuple[int, int],
    global_norm : List[Callable] | None = None,
    preproc_fns: List[Callable] | None = None,
    batch_size: int | None = None,
    n_jobs: int = 1,
    *,
    # The two arguments below are **optional** and are only needed
    # when you run the function with multiprocessing and your model is
    # not picklable (e.g. a Keras model).  They are passed to the internal
    # ProcessPoolExecutor.
    initializer: Callable | None = None,
    initargs: Tuple[Any, ...] = (),
) -> np.ndarray:
    """
    Staggered‑brick tiling + overlapping accumulation + optional
    multiprocessing.  All patches are **pre‑processed first** (by the
    functions supplied in ``preproc_fns``) and then passed as a *batch* to
    ``model.predict``.  The result has the same shape and dtype as the input
    image.
    """
    print('Reached predict_staggered_mean')
    print(global_norm)
    if global_norm is not None:
        for g in global_norm:
            message("Global Normalisation" , 3)
            print('Normalisation Function : ' , g)
            image = g(image)
    # ------------------------------------------------------------------
    # Helper that merges a worker’s partial canvases into the global ones
    # ------------------------------------------------------------------
    def _merge_into_global(sum_g, cnt_g, sum_part, cnt_part):
        sum_g += sum_part
        cnt_g += cnt_part

    # ------------------------------------------------------------------
    # 0️⃣  Basic sanity checks (unchanged)
    # ------------------------------------------------------------------
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2‑D (H,W) or 3‑D (H,W,C).")
    if len(chunk_shape) != 2:
        raise ValueError("chunk_shape must be a 2‑tuple (height,width).")
    if len(overlap) != 2:
        raise ValueError("overlap must be a 2‑tuple (overlap_h,overlap_w).")
    # if n_jobs < 1:
        # raise ValueError("n_jobs must be >= 1.")

    H, W = image.shape[:2]
    step_h, step_w = chunk_shape
    ov_h, ov_w = overlap
    stride_h = step_h - ov_h
    stride_w = step_w - ov_w

    # ------------------------------------------------------------------
    # 1️⃣  Serial version (n_jobs == 1)
    #     pre‑processing before prediction
    # ------------------------------------------------------------------
    if n_jobs == 1:
        # print('Insifr serial version')
        message("Processing and predicting in Serial processing", 2)
        # --------------------------------------------------------------
        #  Gather **all** patches, apply preproc, then batch‑predict
        # --------------------------------------------------------------
        raw_patches = []          # list of patches *before* any preprocessing
        coords      = []          # (r0,c0) for each patch

        total_rows = (H - step_h) // stride_h + 1
        message(f"Generating Row-wise Patched : {total_rows} rows" , 3)
        for row_idx in tqdm(range(total_rows)):
            r0 = row_idx * stride_h
            half_shift = (stride_h // 2) if row_idx % 2 else 0
            c = 0
            while c < W:
                c0 = c + half_shift
                if c0 + step_w > W:
                    break
                patch = image[r0:r0 + step_h, c0:c0 + step_w]
                raw_patches.append(patch)
                coords.append((r0, c0))
                c += stride_w
        message(f"Total Patches generated : {len(raw_patches)}" , 1)
        # --------------------------------------------------------------
        # 2️⃣  Apply custom preprocessing **before** batching
        # --------------------------------------------------------------
        message(f"Applying Local preprocessing to each patch" , 3)
        # TODO : this step is memory intensive, write batch mode for this preprocessing
        if preproc_fns:
            processed = []
            for patch in tqdm(raw_patches):
                for fn in preproc_fns:
                    patch = fn(patch)
                processed.append(patch)
            raw_patches = processed

        # --------------------------------------------------------------
        # 3️⃣  Stack into a batch (add channel dim if needed)
        # --------------------------------------------------------------
        batch = np.stack(raw_patches, axis=0).astype('float32')
        # return batch
        # if batch.ndim == 3:               # (N, H, W) → (N, H, W, 1)
        #     batch = batch[..., np.newaxis]

        # --------------------------------------------------------------
        # 4️⃣  Batched prediction (split into sub‑batches if requested)
        # --------------------------------------------------------------
        # print(batch.shape)
        message(f"Running the Model on each patch" , 3)

        if batch_size is None:
            preds = model.predict(batch, verbose=1).squeeze()
        else:
            chunks = []
            message("Applying model in batches", 'p')
            for start in tqdm(range(0, batch.shape[0], batch_size)):
                sub = batch[start:start + batch_size]
                p   = model.predict(sub, verbose=1).squeeze()
                chunks.append(p)
            preds = np.concatenate(chunks, axis=0)

        # --------------------------------------------------------------
        # 5️⃣  Write predictions back into the *global* canvases
        # --------------------------------------------------------------
        sum_canvas = np.zeros_like(image, dtype=np.float64)
        cnt_canvas = np.zeros_like(image, dtype=np.float64)

        if preds.ndim == 4:               # (N, h, w, C)
            preds = preds.squeeze(axis=-1)

        for (r0, c0), pred in zip(coords, preds):
            sum_canvas[r0:r0 + step_h, c0:c0 + step_w] += pred
            cnt_canvas[r0:r0 + step_h, c0:c0 + step_w] += 1.0

        # --------------------------------------------------------------
        # 6️⃣  Final averaging & cropping (identical to the original code)
        # --------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            merged = sum_canvas / cnt_canvas
        out = merged[:H, :W]

        if np.issubdtype(image.dtype, np.integer):
            out = np.rint(out).astype(image.dtype)
        else:
            out = out.astype(image.dtype)
        return out

    # ------------------------------------------------------------------
    # 2️⃣  Multiprocessing version – rows are split among workers
    # ------------------------------------------------------------------
    else:
        # --------------------------------------------------------------
        # Helper that each worker will execute (very similar to the
        # previous version, but **preproc_fns are applied before prediction**)
        # --------------------------------------------------------------
        print('Inside Parallel Version')
        def _worker_task(rows_slice: Tuple[int, int]):
            """Runs in a child process – same logic as the internal worker
            defined at the top of the file, only with a tiny change:
            pre‑processing happens before the batched predict."""
            # ---- 1️⃣  Collect raw patches belonging to these rows ----
            raw = []
            coords_local = []
            total_rows = (H - step_h) // stride_h + 1
            for row_idx in range(rows_slice[0], rows_slice[1]):
                r0 = row_idx * stride_h
                half_shift = (stride_h // 2) if row_idx % 2 else 0
                c = 0
                while c < W:
                    c0 = c + half_shift
                    if c0 + step_w > W:
                        break
                    raw.append(image[r0:r0 + step_h, c0:c0 + step_w])
                    coords_local.append((r0, c0))
                    c += stride_w

            if not raw:      # nothing to do in this slice
                return (np.zeros_like(image, dtype=np.float64),
                        np.zeros_like(image, dtype=np.float64))

            # ---- 2️⃣  Apply the custom pre‑processing functions ----
            if preproc_fns:
                processed = []
                for patch in raw:
                    for fn in preproc_fns:
                        patch = fn(patch)
                    processed.append(patch)
                raw = processed

            # ---- 3️⃣  Stack into a batch (add channel dim if needed) ----
            batch = np.stack(raw, axis=0).astype('float32')
            if batch.ndim == 3:
                batch = batch[..., np.newaxis]

            # ---- 4️⃣  Batched inference (sub‑batching optional) ----
            if batch_size is None:
                preds = model.predict(batch, verbose=0).squeeze()
            else:
                chunks = []
                for start in range(0, batch.shape[0], batch_size):
                    sub = batch[start:start + batch_size]
                    p   = model.predict(sub, verbose=0).squeeze()
                    chunks.append(p)
                preds = np.concatenate(chunks, axis=0)

            # ---- 5️⃣  Write predictions into the worker‑local canvases ----
            sum_local = np.zeros_like(image, dtype=np.float64)
            cnt_local = np.zeros_like(image, dtype=np.float64)

            if preds.ndim == 4:                # (N, h, w, C)
                preds = preds.squeeze(axis=-1)

            for (r0, c0), pred in zip(coords_local, preds):
                sum_local[r0:r0 + step_h, c0:c0 + step_w] += pred
                cnt_local[r0:r0 + step_h, c0:c0 + step_w] += 1.0

            return sum_local, cnt_local

        # --------------------------------------------------------------
        #  Split the rows into (roughly) equal slices for the workers
        # --------------------------------------------------------------
        # n_workers = max(1, (_n_workers if _n_workers is not None else mp.cpu_count() - 1))

        total_rows = (H - step_h) // stride_h + 1
        if n_jobs == -1: n_jobs = mp.cpu_count() - 2
        rows_per_worker = max(1, total_rows // n_jobs)

        row_slices = []
        cur = 0
        while cur < total_rows:
            nxt = min(cur + rows_per_worker, total_rows)
            row_slices.append((cur, nxt))
            cur = nxt


        n_rows = len(row_slices)
        # arg_list = 
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            futures = [executor.submit(_worker_process_rows, rs) 
                       for rs in zip(
                           row_slices , 
                           n_rows*[image],
                           n_rows*[model] , 
                           n_rows*[chunk_shape] , 
                           n_rows*[overlap], 
                           n_rows*[preproc_fns], 
                           n_rows*[batch_size]
                           )]

            # Global accumulation canvases
            sum_global = np.zeros_like(image, dtype=np.float64)
            cnt_global = np.zeros_like(image, dtype=np.float64)

            for fut in as_completed(futures):
                part_sum, part_cnt = fut.result()
                sum_global += part_sum
                cnt_global += part_cnt

        # --------------------------------------------------------------
        #  Final averaging & cropping (identical to the serial version)
        # --------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            merged = sum_global / cnt_global
        out = merged[:H, :W]

        if np.issubdtype(image.dtype, np.integer):
            out = np.rint(out).astype(image.dtype)
        else:
            out = out.astype(image.dtype)
        return out

from .utility_v2 import load_object

from .ML_utility import get_model_from_weights
import pathlib

def to_abs_path(p: pathlib.Path) -> pathlib.Path:
    """
    Convert *p* (str, os.PathLike or pathlib.Path) to an absolute
    ``Path`` object, resolved against the **current working directory**.

    - ``~`` is expanded.
    - Symbolic links are resolved (``Path.resolve()``).
    - If *p* is already absolute, it is returned unchanged (except for
      ``expanduser`` / ``resolve`` which normalises the path).

    Raises
    ------
    ValueError
        If *p* cannot be interpreted as a path.
    """
    if isinstance(p, pathlib.Path):
        path_obj = p
    elif isinstance(p, (str, pathlib.Path)):
        path_obj = pathlib.Path(p)
    else:
        raise ValueError(f"Unsupported path type: {type(p)!r}")

    # Expand ~ and resolve relative parts against cwd.
    return path_obj.expanduser().resolve()

from .modifiers import modifiers_dict


from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent 

PRED_PATH = str(  BASE_DIR / "../predictors" )
# print('--------------------------------')
print(str(PRED_PATH))

predictors_dict = {
    "HGBS" : {
        'processor':f'{PRED_PATH}/HGBS-C_64v2_trial-5-WBCE-processor-v2.dill', 
        'model-weights' : f'{PRED_PATH}/HGBS_C_64v2_trial-5-WBCE-weights.h5'
    }, 
    "HiGAL" : {
        'processor':f'{PRED_PATH}/HiGAL-C_64v2_trial-5-WBCE-processor-v2.dill', 
        'model-weights' : f'{PRED_PATH}/HiGAL_C_64v2_trial-5-WBCE-weights.h5'
    }
}



@st.cache_data
@st.cache_resource
class filamentIdentifier():
    """
    Wrapper that loads a pre‑trained filament‑segmentation model and provides a
    convenient ``predict`` method for whole‑image inference.

    Parameters
    ----------
    predictor_name : str
        Key in :data:`predictors_dict` that selects the model configuration.
        The entry must contain the fields ``'processor'`` (path to the
        preprocessing pipeline) and ``'model-weights'`` (path to the model
        weights file).

    model_weights : str, optional
        Path to a weights file that overrides the ``'model-weights'`` entry
        from ``predictors_dict[predictor_name]``.  If ``None`` the path from the
        dictionary is used.

    Attributes
    ----------
    local_mods : list[callable]
        List of local (per‑chunk) preprocessing functions extracted from the
        saved processor object.

    global_mods : list[callable]
        List of global (image‑wide) preprocessing functions extracted from the
        saved processor object.

    model : keras.Model
        The TensorFlow/Keras model instantiated with the appropriate input
        shape and loaded with the supplied weights.

    Methods
    -------
    predict(cd, batch_size=None, window_overlap_frac=0.75)
        Run inference on a ``cd`` image container, returning a probability map.
    """
    def __init__(self, predictor_name , model_weights = None):
        """
        Initialise the identifier by loading the preprocessing pipeline and
        the model weights.

        Notes
        -----
        * ``predictors_dict`` and ``modifiers_dict`` are expected to be defined
          elsewhere in the package.
        * ``load_object`` must be able to deserialize the processor object
          (e.g., a ``dill`` file).
        * ``get_model_from_weights`` creates a model with the correct input size.
        """
        prediction_model_name = predictors_dict[predictor_name]['processor']
        model_weights = predictors_dict[predictor_name]['model-weights']
        # identifier = load_object('../predictor/HGBS-predictor-v6.dill')
        '''
        In the save dataprocesors, only the modifier names are stored.
        Builds the data-preprocessing pipeline from dictionary of modifiers available in modifiers.py
        '''
        # Load the saved processor (contains normalisation pipelines)
        proc = load_object(to_abs_path(prediction_model_name))

        # Build local and global normalisation pipelines
        self.local_mods = [modifiers_dict[l] for l in proc.local_normalizer]
        self.global_mods = [modifiers_dict[l] for l in proc.global_normalizer]

        # Instantiate model and load weights
        #     # TODO : fix-this to load any model and not just weights
        model = get_model_from_weights(proc.chunk_params['size'])
        model.load_weights(to_abs_path(model_weights))
        self.model = model

    def predict(self, cd, batch_size=None , window_overlap_frac = 0.75, n_jobs = 1):
        """
        Perform inference on the column density data contained in ``cd``.

        Parameters
        ----------
        cd : object
            An image container that exposes the raw data via ``cd.data``.
            The container type is not enforced; any object with a ``data``
            attribute convertible to a NumPy array is accepted.

        batch_size : int, optional
            Number of chunks to process in parallel per batch.  If ``None`` the
            default batch size of ``process_image_staggered_mean`` is used.

        window_overlap_frac : float, default=0.75
            Fraction of the chunk dimension that will be overlapped between
            neighbouring windows.  Controls the amount of smoothing at chunk
            borders.

        Returns
        -------
        numpy.ndarray
            Probability (or score) map of the same spatial dimensions as the
            input image.

        Raises
        ------
        ValueError
            If the model input shape cannot be inferred or if ``cd`` does not
            provide ``data``.

        Notes
        -----
        The function uses :func:`process_image_staggered_mean` to tile the
        image, apply the local normalisation functions, run the model on each
        tile, and then re‑assemble the results using a mean of overlapping
        regions.
        """

        # Determine the spatial size of a single model input tile
        chunk_shape = self.model.input_shape[1:-1]          # (H, W, C) → (H, W)
        ovlp = int(chunk_shape[0] * window_overlap_frac)   # symmetric overlap
        overlap = (ovlp, ovlp)
        # Run tiled inference
        pi = process_image_staggered_mean(
            image=cd.data,
            model=self.model,
            preproc_fns=self.local_mods,
            global_norm=self.global_mods,
            chunk_shape=chunk_shape,
            overlap=overlap,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )
        return pi






def load_predictor(predictor_name: str) -> Any:
    """Deserialize the predictor object from disk (cached)."""
    predictor = filamentIdentifier(prediction_model_name = predictors_dict[predictor_name]['processor'] , model_weights = predictors_dict[predictor_name]['model-weights'])
    return predictor