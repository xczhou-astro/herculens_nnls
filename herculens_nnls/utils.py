import numpy as np
import jax
from astropy.io import fits

class Tee:
    """Class to duplicate output to both stdout and a file"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def json_serializer(obj):

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, jax.Array):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def center_crop(image, crop_size):

    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    h, w = image.shape[:2]
    start_y = max((h - crop_h) // 2, 0)
    start_x = max((w - crop_w) // 2, 0)
    end_y = start_y + crop_h
    end_x = start_x + crop_w

    return image[start_y:end_y, start_x:end_x]


def get_fits_data(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64)


def _pytree_flat_param_labels(params_pytree):
    """
    Build flat parameter labels that match ravel_pytree() order.
    Scalars keep their site name; vectors/tensors get index suffixes, e.g. a[3].
    """
    try:
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params_pytree)
    except Exception:
        path_leaves = []
        for key in sorted(params_pytree.keys()):
            path_leaves.append(((key,), params_pytree[key]))

    def _entry_to_token(entry):
        if hasattr(entry, "key"):
            return str(entry.key)
        if hasattr(entry, "idx"):
            return str(entry.idx)
        if hasattr(entry, "name"):
            return str(entry.name)
        return str(entry)

    labels = []
    for path, leaf in path_leaves:
        if isinstance(path, tuple):
            base = ".".join(_entry_to_token(p) for p in path)
        else:
            base = _entry_to_token(path)
        arr = np.asarray(leaf)
        if arr.ndim == 0:
            labels.append(base)
        else:
            for idx in np.ndindex(arr.shape):
                idx_txt = ",".join(str(i) for i in idx)
                labels.append(f"{base}[{idx_txt}]")
    return labels


def print_emcee_parameter_uncertainties(flat_samples, init_params):
    """
    Print median and asymmetric 1-sigma uncertainties for each emcee parameter:
      p50 - (p50-p16) + (p84-p50)
    """
    if flat_samples is None or len(flat_samples) == 0:
        print("[emcee] No samples available to compute uncertainties.")
        return

    q16, q50, q84 = np.percentile(flat_samples, [16, 50, 84], axis=0)
    err_lo = q50 - q16
    err_hi = q84 - q50

    labels = _pytree_flat_param_labels(init_params)
    ndim = flat_samples.shape[1]
    if len(labels) != ndim:
        labels = [f"param_{i}" for i in range(ndim)]

    print("[emcee] Parameter posterior summary (50th, -16th, +84th):")
    for i, name in enumerate(labels):
        print(f"  {name}: {q50[i]:.6g} -{err_lo[i]:.6g} +{err_hi[i]:.6g}")


