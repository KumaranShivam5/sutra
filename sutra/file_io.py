"""IO utilities - FITS loading with caching."""

from __future__ import annotations

import numpy as np
import streamlit as st
from astropy.io import fits
from typing import Union, BinaryIO
import copy
import io as sysio
# @st.cache_data
# @st.cache_resource
def load_fits(_file: Union[str, BinaryIO]) -> np.ndarray:
    """
    Load a FITS file (either a path or an uploaded file object) and return the
    primary HDU data as a NumPy 2-D array.

    Parameters
    ----------
    file : str | BinaryIO
        Path to a FITS file or an uploaded file-like object.

    Returns
    -------
    np.ndarray
        2-D image data.
    """
    if isinstance(_file, str):
        hdul = fits.open(_file , ignore_missing_simple=True)
    else:
        # Streamlit uploads provide a BytesIO-like object
        hdul = fits.open(_file,  ignore_missing_simple=True)
    # data = copy.deepcopy(hdul[0])
    # hdul.close()
    return hdul[0]


@st.cache_data
def download_fits(data, _header):
    ff = fits.HDUList([fits.PrimaryHDU(data = data, header=_header)])
    fits_buffer = sysio.BytesIO()
    ff.writeto(fits_buffer)
    fits_buffer.seek(0)
    return fits_buffer


def download_fits_non_st(data, _header):
    ff = fits.HDUList([fits.PrimaryHDU(data = data, header=_header)])
    return ff
    # return fits_buffer