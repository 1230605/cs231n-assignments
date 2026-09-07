"""Utility functions used for viewing and processing images."""

import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from imageio import imread
from PIL import Image



def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array (H, W, 3) uint8.
    If the URL is unreachable (common for the old Flickr links), return a
    gray placeholder so that downstream visualization code still runs.
    """
    import io, time

    try:
        with urllib.request.urlopen(url) as f:
            data = f.read()
        _, fname = tempfile.mkstemp()
        try:
            with open(fname, "wb") as ff:
                ff.write(data)
            with open(fname, "rb") as fh:
                img = np.array(Image.open(io.BytesIO(fh.read())))
        finally:
            for _ in range(20):
                try:
                    os.remove(fname)
                    break
                except PermissionError:
                    time.sleep(0.05)
        return img
    except Exception as e:
        print("Image URL failed (", type(e).__name__, "), using gray placeholder:", url)
        return np.full((128, 128, 3), 128, dtype=np.uint8)



def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: the width, height values are currently flipped here, and we should
        # change the resampling method to BILINEAR to match the torch implementation
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
