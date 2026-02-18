"""Filament characterisation utilities."""

from __future__ import annotations
# from email import message

from sutra.logger import message

import numpy as np
import pandas as pd
import streamlit as st
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

from sutra.profiler.profiling import RadProf



# @st.cache_data
# def characterise_filaments(
#     cd_map: np.ndarray,
#     skeleton: np.ndarray,
#     smoothing_frac: float,
#     distance_pc: float,
#     beam_size: float,
# ) -> pd.DataFrame:
#     """
#     Compute simple filament properties.

#     Parameters
#     ----------
#     cd_map : np.ndarray
#         Column-density map.
#     skeleton : np.ndarray
#         Binary skeleton map.
#     smoothing_frac : float
#         Fraction for Gaussian smoothing (0-1).
#     distance_pc : float
#         Distance to the cloud in parsec.
#     beam_size : float
#         Telescope beam size in arcseconds.

#     Returns
#     -------
#     pd.DataFrame
#         One row per filament with length, width, mean column density, etc.
#     """
#     # 1 Smooth CD map (sigma derived from fraction of image size)
#     sigma = smoothing_frac * max(cd_map.shape) / 2.0
#     smooth_cd = gaussian_filter(cd_map, sigma=sigma)

#     # 2 Label individual skeleton branches
#     labelled = label(skeleton)
#     props = regionprops(labelled, intensity_image=smooth_cd)

#     rows = []
#     for p in props:
#         length_pix = p.perimeter
#         # Convert pixel length to physical length (pc) using distance & beam
#         # Approximate pixel scale: beam_size (arcsec) -> pc at given distance
#         # 1 rad = 206265 arcsec, so pixel_scale_pc = distance_pc * (beam_size / 206265)
#         pixel_scale_pc = distance_pc * (beam_size / 206265.0)
#         length_pc = length_pix * pixel_scale_pc

#         mean_cd = p.mean_intensity
#         rows.append(
#             {
#                 "filament_id": p.label,
#                 "length_pc": length_pc,
#                 "mean_column_density": mean_cd,
#                 "pixel_count": p.area,
#             }
#         )

#     df = pd.DataFrame(rows)
#     return df

from sutra.profiler.profiling import RadProf
# from sutra.profiler.Filament import Field

from sutra.profiler.PlotFil import plot_fil

# @st.cache_data
def skeleton(field, prob_thresh = 0.5):
    field.filter_background()
    field.run_skel(prob_thresh, prune=True)
    field.tangents(10,5)
    field.reorder(3)
    field.spline_smooth(update = True)
    return field

# @st.cache_data
def apply_skl_threshold(field , prob_thresh = 0.5):
    field.filter_background()
    field.run_skel(prob_thresh, prune=True)
    return field.skel

# @st.cache_resource
def find_profiles(cd_map , skeleton , meta_info):
    field = RadProf()
    field.tangents(10,5)
    field.reorder(3)
    field.spline_smooth(update = True)
    return field



from sutra.profiler.Filament import propsMap

# @st.cache_resource
class local_field():
    def __init__(_self, _cd_map , skeleton , meta_info):
        _self.radprof = RadProf(img = _cd_map , mask = skeleton , meta_info = meta_info)
        _self.fil_table = None
        _self.fil_prop_map = None
        _self.fil_prop_map_table = None
    
    # @st.cache_data
    def apply_skl_threshold(_self, th):
        print('running function apply_skl_threshold')
        _self.radprof.filter_background()
        _self.radprof.run_skel(th, prune=True)
    
    # @st.cache_data
    def process_skeleton(_self, ks=7, stride=5 , reorder=4):
        _self.radprof.tangents(ks = ks,stride = stride , )
        _self.radprof.reorder(reorder)
        _self.radprof.spline_smooth(update = True)
    
    def extract_rad_prof(_self, N_CUT_OFF_PIX = 50):
        _self.radprof.cut_off_points(50)
        # _self.radprof.create_rad_profile()
        _self.radprof.create_rad_profile_single_thread()
    
    # @st.cache_data
    def plot_fil(_self, indx):
        print('again plotting------')
        fig = plot_fil(_self.radprof, indx , get_fig = True)
        return fig
    # def set_Field(_self):
        # self._Field = Field(_self.radprof)
    def get_filament_table(_self, stride = 1, refresh=False, ):
        message("BP2", 3)
        print('BP2')
        if _self.fil_table is None or refresh:
            # pritn('BP 1')
            message("BP1", 3)
            all_filament_map = propsMap(_self.radprof)
            # _self.prop_field = all_filament_map
            _self.fil_prop_map = all_filament_map
            all_filament_map.collect_filaments(stride = stride)
            _self.filament_collection = all_filament_map
            musca_fil_table =  all_filament_map.filament_table()
            # musca_fil_table = musca_fil_table.drop(columns = ['Location'])
            _self.fil_table = musca_fil_table
            return musca_fil_table
            # charview.dataframe(musca_fil_table)
        else: return _self.fil_table
    
    def get_filament_prop_map(self, stride = 1 , refresh = False):
        print("[INFO]-- computing filament properties as PROP-MAP" )
        if self.fil_prop_map is None:
            self.fil_prop_map = propsMap(self.radprof)
        if self.fil_prop_map_table is None or refresh:
            print("Refreshing...")
            print(refresh)
            self.fil_prop_map_table = self.fil_prop_map.get_props_maps(stride = stride, refresh=refresh)
        return self.fil_prop_map_table