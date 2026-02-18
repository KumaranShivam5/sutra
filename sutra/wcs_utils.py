
from astropy.wcs import WCS 
from astropy import units as u
import streamlit as st

# @st.cache_data
def _get_beam_size_pixel_scale(pixel_scale, beam_arcsec):
    """
    support function for streamlit caching
    """
    ps = u.pixel_scale((pixel_scale*u.degree/u.pixel)) # convert beamsize to pixel size
    resolution = beam_arcsec*u.arcsec # assuming resolution is 36.4 arcsec (not to be confuse with pixel size, it is beam size)
    hpbw = resolution.to(u.pixel, ps) # convert resolution to pixel coordinate
    pixel_size = resolution.to(u.pixel, ps)
    return hpbw , pixel_size


def get_pixel_to_arcsec(header , pixel):
    wcs = WCS(header)
    pixel_scale = wcs.pixel_scale_matrix[1,1]*60*60
    # ps = u.pixel_scale((pixel_scale*u.degree/u.pixel)) # convert beamsize to pixel size
    return pixel*pixel_scale

def get_beam_size(header, beam_arcsec):
    '''
    Obtain beam size in arcsec and in pixel for a given WCS header

    Parameters
    ----------
    header : fits.header
        FITS HDU header.
    beam_arcsec : float
        beam size in arcsec (without units)

    Returns
    -------
    (float , float)
        size of beam in terms of pixel & 
        size of one pixel in arcsec unit
    """
    '''
    wcs = WCS(header)
    hpbw , pixel_size = _get_beam_size_pixel_scale(wcs.pixel_scale_matrix[1,1], beam_arcsec)
    return hpbw , pixel_size


# @st.cache_data
def _get_sky_dist_pixel_scale(pixel_scale, pixdist, distance):
    ps = u.pixel_scale((pixel_scale*u.degree/u.pixel)) 
    tdist = (pixdist*u.pixel).to(u.rad, ps)*distance
    tdist = [t.value for t in tdist]
    return tdist

def get_sky_dist(header, pixdist, distance):
    wcs = WCS(header)
    tdist = _get_sky_dist_pixel_scale(wcs.pixel_scale_matrix[1,1], pixdist, distance)
    return tdist