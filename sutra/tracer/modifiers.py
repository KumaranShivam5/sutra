import numpy as np

from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction, dilation
from scipy.ndimage import standard_deviation, mean

def remove_nan(arr):
    return np.nan_to_num(arr, 0)

from skimage.morphology import dilation
def dialate(arr):
    return dilation(arr)

def zero_one_norm(arr):
    arr = np.asarray(arr, dtype=float)
    max_val = np.nanmax(arr)
    if np.isnan(max_val):
        # Nothing to scale – return an array full of NaNs
        return np.full_like(arr, np.nan, dtype=float)
    if max_val == 0:
        # Division would be 0/0 → return zeros (preserve NaNs)
        out = np.zeros_like(arr, dtype=float)
        out[np.isnan(arr)] = np.nan
        return out
    with np.errstate(invalid='ignore', divide='ignore'):
            out = arr / max_val
    return out
    # return arr / (np.nanmax(arr))

def fill_nan(thresh):
    def _fill_nan(arr):
        arr[arr<thresh] = np.nan
        return remove_nan(arr)
    return _fill_nan

def good_pixel_filter(arr):
    import numpy as np
    arr[arr<0] = np.nan
    return arr

def std_norm(arr):
   return (arr - mean(arr)) / standard_deviation(arr)  

def histogram_eq(cd):
    img_eq = np.sort(cd.ravel()).searchsorted(cd)
    return img_eq


def flatten(flatten_percent = 95):
    def _flatten(arr):
        thresh_val = np.percentile(arr, flatten_percent)
        return thresh_val * np.arctan(arr / thresh_val)
    return _flatten


# from astropy.io import fits 
# def flatten(flatten_percent=95):
#     from fil_finder import FilFinder2D
#     def flatten_in(arr , flatten_percent=flatten_percent):
#         fil = FilFinder2D(arr)
#         fil.preprocess_image(flatten_percent=flatten_percent)
#         return fil.flat_img.value
#     return flatten_in

# from fil_finder import FilFinder2D

# def flatten(arr):
#     fil = FilFinder2D(arr)
#     fil.preprocess_image(flatten_percent=75)
#     return fil.flat_img.value

def bkg_removal(arr):
    image = img_as_float(arr)
    # image = gaussian_filter(image, 1)
    seed = np.copy(image)
    seed[1:-1,1:-1] = image.min()
    mask = image 
    dialated = reconstruction(seed, mask, method='dilation')
    return np.array(image - dialated)




def target_binary(arr):
    arr[arr>0] = 1 
    arr[arr<=0] = 0 
    return arr

def skl_dilate(arr):
    return np.array(dilation(arr))


from astropy.convolution import Gaussian2DKernel , convolve

# arr =  np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
def blur_line(arr):
    gk = Gaussian2DKernel(x_stddev=2, y_stddev=2)
    return convolve(arr, gk) 


modifiers_dict = {
    "remove_nan" : remove_nan , 
    "dialate" : dialate , 
    "zero_one_norm" : zero_one_norm, 
    "fill_nan" : fill_nan, 
    "good_pixel_filter" : good_pixel_filter,
    "std_norm" : std_norm, 
    "histogram_eq" : histogram_eq,
    "bkg_removal" : bkg_removal, 
    "target_binary" : target_binary,
    "skl_dilate" : skl_dilate, 
    "blur_line" : blur_line, 
    "flatten" : flatten(95),
}
