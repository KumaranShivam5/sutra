

import numpy as np
import streamlit as st
from sutra.logger import message


from scipy.signal import convolve2d
from skimage.morphology import skeletonize 
from skimage.morphology import dilation , ellipse, skeletonize , remove_small_objects

from astropy.convolution import convolve , Gaussian2DKernel , convolve_fft

from skimage.filters import apply_hysteresis_threshold


import copy

@st.cache_data
def filter_background(CD, val=None):
    v = val
    CD = copy.deepcopy(CD)
    CD[CD<=0] = np.nan 
    cd_vals = np.log10(CD.flatten())
    cd_vals[cd_vals == np.inf] = np.nan

    if val is None:
        bin_bnds = np.arange(np.nanmin(cd_vals) , np.nanmax(cd_vals) , 0.2)
        _,v = np.histogram(cd_vals , bins = bin_bnds)
    # return v
        v = v[1]
    bkg = cd_vals[cd_vals < v]
    # return bkg
    bkg_threshold = np.power(10, np.nanmedian(bkg))
    message(f'Creating Background mask | backgroung threshold {bkg_threshold}', 3)
    bkg_mask = CD > bkg_threshold
    return bkg_mask , bkg_threshold
import copy

def convolve_map(array, beam_size):
    kernel_size = beam_size / 2.355 
    beam_kernel = Gaussian2DKernel(kernel_size,kernel_size)
    prob_conv = kernel_size*convolve(array, beam_kernel)
    return prob_conv
# @st.cache_data
def run_skel(prob_map , th_max , th_min = None, beam_size = 12, bkg_mask = None , prune = False , convolve_map = True):
    print('running updated')
    kernel_size = beam_size / 2.355 
    beam_kernel = Gaussian2DKernel(kernel_size,kernel_size)
    prob_conv = copy.deepcopy(prob_map)
    if convolve_map:prob_conv = kernel_size*convolve_fft(prob_map, beam_kernel)
    if th_min: # apply hysteresis threshold if both min and max is given
        skel = apply_hysteresis_threshold(prob_conv , th_min , th_max)
    else: skel = (prob_conv > th_max )*1
    skel = skeletonize(skel)
    skel = dilation(skel,  footprint=ellipse(int(np.around(beam_size / 2)), int(np.around(beam_size/2))))
    skel = skeletonize(skel)
    if prune: skel = remove_small_objects(skel, min_size = beam_size*6, connectivity=beam_size)*1
    if bkg_mask is not None: skel*=(bkg_mask*1)
    return skel


