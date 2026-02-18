import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize 
from skimage.morphology import dilation , ellipse, skeletonize , remove_small_objects
from skimage.draw import line
from skimage.filters import gaussian
from scipy.interpolate import splprep, splev
from astropy.io import fits
# from .astroplot import plot

import pandas as pd



from scipy.signal import argrelmin , argrelmax
import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


# import streamlit as st


def message(msg, type=1):
    tp = {
        1 : '[INFO] >> ' , 
        2 : '>>>>> [DEBUG] >>>>> ' , 
    }
    print(f'{tp[type]} {msg}')
    # st.write(f'{tp[type]} {msg}')
    return None

def deresolve(arr, size, mode='mean'):
    newarr = []
    for i in range(0, len(arr), size):
        if(mode=='mean'):
            newrow = [np.mean(arr[i:i+size, j:j+size])*size for j in range(0, len(arr[0]), size)]
        elif(mode=='max'):
            newrow = [np.max(arr[i:i+size, j:j+size]) for j in range(0, len(arr[0]), size)]
        newarr.append(newrow)

    return(newarr)




import multiprocessing as mp
from typing import List, Tuple

def _worker_tangent_chunk(
    skel: np.ndarray,
    i_start: int,
    i_end: int,
    ks: int,
    stride: int,
) -> Tuple[List[float], List[int], List[int]]:
    m_arr, cen_x, cen_y = [], [], []
    for i in range(i_start, i_end, stride):
        if i > skel.shape[0] - ks:
            break
        for j in range(0, skel.shape[1] - ks, stride):
            box = skel[i:i+ks, j:j+ks]
            if np.sum(box) < 3:
                continue
            inds = np.where(box.ravel())[0]
            x = (inds // ks).astype(int)
            y = inds - ks * x
            coeff, _, r, _, _ = np.polyfit(x + 1, y + 1, 1, full=True)
            if r < 2:
                m_arr.append(0.0)
            else:
                if abs(coeff[0]) < 1e-12:
                    m_arr.append(999.0)
                else:
                    m_arr.append(-1.0 / coeff[0])
            cen_x.append(int(np.around(np.median(x))) + i)
            cen_y.append(int(np.around(np.median(y))) + j)
    return m_arr, cen_x, cen_y


# ----------------------------------------------------------------------
# Worker – runs in a separate process
# ----------------------------------------------------------------------
def _smooth_segment(task):
    """
    Compute the spline‑based smoothing for a single filnum segment.

    Parameters
    ----------
    red_x, red_y : 1‑D ndarray
        Reduced centre coordinates of the segment (without the first and
        last points).
    red_adj      : 1‑D ndarray
        Corresponding adjacency distances.
    seg_id       : int
        Original segment identifier (i in the original loop).
    offset       : int
        Number of previously‑skipped segments (the *minus* value that
        would have been accumulated before this segment).
    max_knots    : int
        Smoothing size – computed from ``self.hpbw.value/3`` in the
        original method.

    Returns
    -------
    tuple containing the data that the original loop appends to the
    various lists:
        (newpt_x, newpt_y, dpt_x, dpt_y,
         red_x, red_y, red_adj,
         newder_x, newder_y,
         seg_label)
    """
    red_x, red_y, red_adj, seg_id, offset, max_knots = task

    # ---- replicate the original per‑segment calculations ----------
    u = np.arange(len(red_x))
    mkf = int(np.around(len(red_x) / max_knots))

    # spline fitting (k=4, nest=mkf)
    tck, _ = splprep([red_x, red_y], k=3 ,u=u , s = 0.5)

    # points on the original parameter grid
    newpt = splev(u, tck, der=0)

    # densely sampled points for the visual smooth skeleton
    dense_u = np.linspace(0, len(red_x), len(red_x) * 5)
    newdpt = splev(dense_u, tck, der=0)

    # first derivative – needed for the new slope (m)
    newder = splev(u, tck, der=1)

    # slope (dy/dx) → later transformed to -1/m
    slope = newder[1] / newder[0]

    # label for this segment after accounting for skipped ones
    seg_label = np.ones(len(red_x)) * (seg_id - offset)

    return (np.array(newpt[0]), np.array(newpt[1]),
            np.array(newdpt[0]), np.array(newdpt[1]),
            red_x, red_y, red_adj,
            slope, seg_label)



def _get_sky_dist(pixdist , distance , pixel_scale_matrix):
    # wcs = WCS(self.header)
    # ps = u.pixel_scale((wcs.pixel_scale_matrix[1,1]*u.degree/u.pixel)) 
    ps = u.pixel_scale((pixel_scale_matrix*u.degree/u.pixel)) 
    tdist = (pixdist*u.pixel).to(u.rad, ps)*distance
    tdist = [t.value for t in tdist]
    return tdist

def _rad_profile_worker(args):
    """
    args = (xl, yl, xh, yh, xc, yc, fn,
            img_shape, img, sky_dist_func_name, sky_dist_kwargs)

    * sky_dist_func_name : string identifying which sky‑distance
                          routine to call (e.g. "linear", "polynomial")
    * sky_dist_kwargs    : optional dict that will be unpacked and
                          passed to the chosen routine.
    """
    (xl, yl, xh, yh, xc, yc, fn,
     img_shape, img, distance , pixel_scale_matrix) = args

    # ---- bounds check (identical to original) ----
    if (min(xl, xh) < 0 or max(xl, xh) > img_shape[0] - 1 or
        min(yl, yh) < 0 or max(yl, yh) > img_shape[1] - 1):
        return None

    # ---- three‑point line (low‑centre‑high) ----
    newx = [xl, xc, xh]
    newy = [yl, yc, yh]

    # ---- Bresenham line (the same `line` used originally) ----
    xx, yy = line(int(xl), int(yl), int(xh), int(yh))
    line_x = np.asarray(xx, dtype=int)
    line_y = np.asarray(yy, dtype=int)

    # ---- column density (vectorised) ----
    col_dens = img[line_x, line_y]

    # ---- pixel distance from the centre point ----
    pt   = line_x.size
    dmin = np.hypot(xl - xc, yl - yc)          # low‑centre distance
    dmax = np.hypot(xh - xc, yh - yc)          # high‑centre distance
    pix_dist = np.linspace(-dmin, dmax, pt)

    # ---- sky distance – choose routine by name ----
    # The actual functions live in the module `sutra.profiler.utils`
    # from sutra.profiler.utils import sky_distance_routines

    # sky_func = sky_distance_routines[sky_dist_func_name]
    # sky_dist = sky_func(pix_dist, **sky_dist_kwargs)
    sky_dist = _get_sky_dist(pix_dist , distance=distance, pixel_scale_matrix=pixel_scale_matrix)
    # ---- dictionary – exactly what the original code stored ----
    prof = {
        'line_x'   : line_x,
        'line_y'   : line_y,
        'cen'      : [xc, yc],
        'low'      : [xl, yl],
        'high'     : [xh, yh],
        'filnum'   : fn,
        'slope'    : np.arctan2((yh - yl), (xh - xl)),
        'col_dens' : col_dens,
        'pix_dist' : pix_dist,
        'sky_dist' : sky_dist
    }

    return (newx, newy, line_x, line_y, prof)




def _dist_chunk(
    chunk: np.ndarray,
    cen_x: np.ndarray,
    cen_y: np.ndarray,
    cur_x: float,
    cur_y: float,
) -> np.ndarray:
    """
    Compute Euclidean distances from the current point (cur_x,cur_y)
    to all points whose indices are in *chunk*.
    """
    dx = cen_x[chunk] - cur_x
    dy = cen_y[chunk] - cur_y
    return np.sqrt(dx * dx + dy * dy)




class RadProf:
    def __init__(self, img, mask=None , meta_info = None):
        self.img = img.data
         # this global threshold should be calculated from the data
        self.mask = mask
        self.header = img.header
        self.hdu = img
        self.skel = None
        self.smooth_skel = None
        self.isnode = []
        # self.distance = distance
        self.meta_info = meta_info
        self.bkg_threshold = None

        wcs = WCS(self.header)
        ps = u.pixel_scale((wcs.pixel_scale_matrix[1,1]*u.degree/u.pixel)) # convert beamsize to pixel size
        resolution = self.meta_info['beam']*u.arcsec # assuming resolution is 36.4 arcsec (not to be confuse with pixel size, it is beam size)
        self.hpbw = resolution.to(u.pixel, ps) # convert resolution to pixel coordinate
        self.pixel_size = resolution.to(u.pixel, ps)

        self.m_arr = []
        self.cen_x = []
        self.cen_y = []
        self.adj_dist = []
        self.filnum = []


        self.x_low = []
        self.y_low = []
        self.x_high = []
        self.y_high = []
        
        self.prof_dict = []

    # def set_cut_off(self , cut_off_size = None):
    def get_sky_dist(self, pixdist):
        wcs = WCS(self.header)
        ps = u.pixel_scale((wcs.pixel_scale_matrix[1,1]*u.degree/u.pixel)) 
        tdist = (pixdist*u.pixel).to(u.rad, ps)*self.meta_info['distance']
        tdist = [t.value for t in tdist]
        return tdist
    # def set_cut_off(self , cut_off_size = None):

    def pc_to_pixel(self , pcdist):
        #conversion factor
        cf = 1 / self.get_sky_dist(pixdist = [1])[0]
        # return cf
        return np.asarray(pcdist)*cf
        

    def filter_background(self, val=None):
        self.img[self.img<=0] = np.nan 
        # global background filtering
        cd_vals = np.log10(self.img.flatten())
        cd_vals[cd_vals == np.inf] = np.nan
        # print(np.nanmin(cd_vals) , np.nanmax(cd_vals) )
        # print(np.nanmin(cd_vals) , np.nanmax(cd_vals) )
        bin_bnds = np.arange(np.nanmin(cd_vals) , np.nanmax(cd_vals) , 0.1)
        _,v = np.histogram(cd_vals , bins = bin_bnds)
        # return v
        bkg = cd_vals[cd_vals < v[1]]
        # return bkg
        bkg_threshold = np.power(10, np.nanmedian(bkg))
        self.bkg_threshold = 1*bkg_threshold
        message(f'Filtering background | backgroung threshold {bkg_threshold}')
        self.global_threshold = bkg_threshold
        # self.img[self.img < bkg_threshold] = np.nan

    def run_skel(self , mask_tol = 0.999, prune = False):
        if(self.mask is not None):
            mask = self.mask > mask_tol
        else: mask = np.ones(self.img.shape)
        mean_kernel = np.ones((4, 4))/16
        mask = convolve2d(mask, mean_kernel, mode='same')
        self.skel = skeletonize(mask)
        self.skel = np.array((self.skel > 0), dtype=int)
        if(self.bkg_threshold is not None):
            print(self.bkg_threshold)
            self.skel[self.img < self.bkg_threshold] = False
            # self.skel[self.img]
            message('Removing background')
        if(prune):
            message('Pruining')
            self.skel =  remove_small_objects(skeletonize(dilation(self.skel , ellipse(3,3)) , ) > 0 , min_size  = 48 , connectivity = 6)
        # message('Check if skeleton exists', 2)
        # print(self.skel)
        # message('Check if skeleton exists', 2)
    
    def set_skeleton(self, skeleton):
        """
        instead of skeleton from the Model output,
        skeleton from any other algorithm can be provided here
        """
        self.skel = skeleton

    def tangents(self, ks=5, stride=5, _n_workers: int | None = None):
        # message('Check if skeleton exists', 2)
        # print(self.skel)
        # message('Check if skeleton exists', 2)

        # message(f'Finding tangents to the skeleton | Pixel distance {stride} | End padding {ks}')
        # m_arr = []
        # cen_x = []
        # cen_y = []
        # for i in range(0, len(self.skel)-ks, stride):
        #     for j in range(0, len(self.skel[0])-ks, stride):
        #         box = self.skel[i:i+ks, j:j+ks]
        #         if(np.sum(box)<3): 
        #             continue
        #         inds = np.where(box.flatten())[0]
        #         x = np.array(inds/ks, dtype='int')
        #         y = inds - ks*x
        #         coeff, _, r, _, _ = np.polyfit(x+1,y+1,1, full=True)
        #         # if(coeff[0]<0.000001): m_arr.append(999)
        #         if(r<2): m_arr.append(0)
        #         else: m_arr.append(-1/coeff[0])

        #         cen_x.append(np.around(np.median(x))+i)
        #         cen_y.append(np.around(np.median(y))+j)

        # self.m_arr = np.array(m_arr)
        # self.cen_x = np.array(cen_x)
        # self.cen_y = np.array(cen_y)
        message(f'Finding tangents to the skeleton in multiprocessing mode | Pixel distance {stride} | End padding {ks}')
        n_workers = max(1, (_n_workers if _n_workers is not None else mp.cpu_count() - 1))

        max_i = self.skel.shape[0] - ks
        slice_edges = np.linspace(0, max_i, n_workers + 1, dtype=int)

        with mp.Pool(processes=n_workers) as pool:
            args = ((self.skel,
                    int(slice_edges[w]),
                    int(slice_edges[w + 1]),
                    ks,
                    stride) for w in range(n_workers))
            partial = pool.starmap(_worker_tangent_chunk, args)

        m_arr, cen_x, cen_y = [], [], []
        for p_m, p_x, p_y in partial:
            m_arr.extend(p_m)
            cen_x.extend(p_x)
            cen_y.extend(p_y)

        self.m_arr = np.array(m_arr, dtype=float)
        self.cen_x = np.array(cen_x, dtype=int)
        self.cen_y = np.array(cen_y, dtype=int)
        message(f'Tangent finding completed')

    def nodes(self, blur = True, sigma=1):
        message('Finding Nodes')
        isnode = np.zeros(self.img.shape)
        for i in range(0, len(self.skel)-2, 1):
            for j in range(0, len(self.skel[0])-2, 1):
                box = self.skel[i:i+2, j:j+2]
                if(np.sum(box)>2): 
                    isnode[i+1, j+1] = 1

        if(blur):        
            tmp = gaussian(isnode, sigma = sigma)
            self.isnode = tmp>0.0001
        else: 
            self.isnode = isnode

    def reorder(self, count_thres=1):
        message("Reordring")
        count_arr = []
        strt = np.argmin(self.cen_x)
        count_arr.append(strt)
        ind_arr = np.arange(0, len(self.cen_x))
        adj_dist =[0.0]
        while len(count_arr)<len(self.cen_x):
            tempinds = np.setdiff1d(ind_arr, count_arr)
            distarr = np.sqrt((self.cen_x[tempinds] - self.cen_x[strt])**2 + (self.cen_y[tempinds] - self.cen_y[strt])**2)
            strt = tempinds[np.argmin(distarr)]
            count_arr.append(strt)
            adj_dist.append(distarr[np.argmin(distarr)])

        self.m_arr = self.m_arr[count_arr]
        self.cen_x = self.cen_x[count_arr]
        self.cen_y = self.cen_y[count_arr]
        self.adj_dist = np.array(adj_dist)

        filnum_arr = []
        filcount = 0
        for i in range(len(self.m_arr)):
            if(self.adj_dist[i]>count_thres): filcount+=1
            filnum_arr.append(filcount)

        self.filnum = np.array(filnum_arr, dtype='int')
        message("Reordering Done")

    def spline_smooth(self, update=True, max_knots = None):
        # if max_knots is not None
        max_knots = int(self.hpbw.value)
        message(f'Smoothening skeleton | smooth size : {max_knots}')
        spline_x = []
        spline_y = []
        filt_x = []
        filt_y = []
        filt_adj = []
        newnumarr = []
        newm = []
        smoothskel_x = []
        smoothskel_y = []
        minus = 0
        for i in range(0, self.filnum[-1]+1):
            red_x = self.cen_x[self.filnum == i][1:-1]
            red_y = self.cen_y[self.filnum == i][1:-1]
            red_adj = self.adj_dist[self.filnum == i][1:-1]
            if(len(red_x)<5): 
                minus+=1
                continue

            u = np.arange(0, len(red_x))
            mkf = int(np.around(len(red_x)/max_knots))

            # tck, uf = splprep([red_x, red_y], k = 4, u=u, nest=mkf)
            tck, uf = splprep([red_x, red_y], k = 3, u = u, s = 0.5)
            newpt = splev(u, tck, der=0)
            newdpt = splev(np.linspace(0, len(red_x), len(red_x)*5), tck, der=0)
            newder = splev(u, tck, der=1)

            spline_x.append(newpt[0])
            spline_y.append(newpt[1])

            smoothskel_x.append(newdpt[0])
            smoothskel_y.append(newdpt[1])

            filt_x.append(red_x)
            filt_y.append(red_y)
            filt_adj.append(red_adj)
            newm.append(newder[1]/newder[0])

            # print(len(newpt[0]), len(red_x))

            newnumarr.append(np.ones(len(red_x))*i - minus)

        spline_x = np.concatenate(spline_x, axis=0)
        spline_y = np.concatenate(spline_y, axis=0)
        smoothskel_x = np.concatenate(smoothskel_x, axis=0)
        smoothskel_y = np.concatenate(smoothskel_y, axis=0)
        filt_x = np.concatenate(filt_x, axis=0)
        filt_y = np.concatenate(filt_y, axis=0)
        filt_adj = np.concatenate(filt_adj, axis=0)
        newnumarr = np.concatenate(newnumarr, axis=0)
        newm = np.concatenate(newm, axis=0)
        newm = -1/newm

        if(update==True):
            self.m_arr = newm
            self.cen_x = spline_x
            self.cen_y = spline_y
            self.adj_dist = filt_adj
            self.filnum = np.array(newnumarr, dtype='int')

        smoothskel_x = np.array(smoothskel_x, dtype='int')
        smoothskel_y = np.array(smoothskel_y, dtype='int')
        img_dum = np.zeros_like(self.skel)
        xind = (smoothskel_x < img_dum.shape[0])*(smoothskel_x > 0 )
        yind = (smoothskel_y < img_dum.shape[1])*(smoothskel_y > 0)
        ind = [xi and yi for xi,yi in zip(xind , yind)]
        # smoothskel_x = smoothskel_x[smoothskel_x<img_dum.shape[0]]
        # smoothskel_y = smoothskel_y[smoothskel_y<img_dum.shape[1]]
        # print(smoothskel_x.shape)
        # print(smoothskel_y.shape)
        # return smoothskel_x , smoothskel_y
        smoothskel_x = smoothskel_x[ind]
        smoothskel_y = smoothskel_y[ind]
        # return smoothskel_x , smoothskel_y
        img_dum[smoothskel_x, smoothskel_y] = 1
        self.smooth_skel = img_dum

        message('Smooth done')


        return(spline_x, spline_y, filt_x, filt_y, newnumarr, newm)

    def spline_smooth_multiprocess(self, update=True, max_knots=None):
        """
        Smooth the skeleton using B‑spline interpolation.
        The public signature (self, update=True, max_knots=None) is kept
        identical to the original implementation; only the per‑segment
        processing loop is parallelised.
        """
        # ------------------------------------------------------------------
        # 1️⃣  Determine smoothing size (identical to original code)
        # ------------------------------------------------------------------
        max_knots = int(self.hpbw.value / 3)
        message(f'Smoothening skeleton with m ultiprocessing | smooth size : {max_knots}')

        # ------------------------------------------------------------------
        # 2️⃣  Gather data for every filnum segment
        # ------------------------------------------------------------------
        last_seg = int(self.filnum[-1])
        seg_ids = np.arange(last_seg + 1)

        # store temporary per‑segment data
        red_x_list = []
        red_y_list = []
        red_adj_list = []
        valid_mask = []

        for i in seg_ids:
            mask = self.filnum == i
            # strip first and last points (as original code does)
            rx = self.cen_x[mask][1:-1]
            ry = self.cen_y[mask][1:-1]
            radj = self.adj_dist[mask][1:-1]

            red_x_list.append(rx)
            red_y_list.append(ry)
            red_adj_list.append(radj)
            valid_mask.append(len(rx) >= 5)

        red_x_arr = np.array(red_x_list, dtype=object)
        red_y_arr = np.array(red_y_list, dtype=object)
        red_adj_arr = np.array(red_adj_list, dtype=object)
        valid_mask = np.array(valid_mask, dtype=bool)

        # ------------------------------------------------------------------
        # 3️⃣  Compute the “minus” offset for each segment (how many previous
        #     segments were discarded because they were too short)
        # ------------------------------------------------------------------
        # cumulative count of discarded segments up to *and including* each index
        skipped_cumsum = np.cumsum(~valid_mask)
        # offset for a valid segment = number of skipped segments before it
        offset = skipped_cumsum - (~valid_mask)          # remove current if it is skipped
        # new segment label after accounting for skips
        new_labels = seg_ids - offset

        # ------------------------------------------------------------------
        # 4️⃣  Build the list of tasks for the pool (only the valid ones)
        # ------------------------------------------------------------------
        tasks = []
        for i, is_ok in enumerate(valid_mask):
            if not is_ok:
                continue                      # will be omitted exactly as in the original loop
            tasks.append((red_x_arr[i], red_y_arr[i], red_adj_arr[i],
                        i, offset[i], max_knots))

        # ------------------------------------------------------------------
        # 5️⃣  Parallel processing of the valid segments
        # ------------------------------------------------------------------
        n_workers = max(1, mp.cpu_count() - 1)
        pool = mp.Pool(processes=n_workers)

        results = pool.map(_smooth_segment, tasks)

        pool.close()
        pool.join()

        # ------------------------------------------------------------------
        # 6️⃣  Unpack and concatenate results exactly as the original code did
        # ------------------------------------------------------------------
        spline_x, spline_y = [], []
        smoothskel_x, smoothskel_y = [], []
        filt_x, filt_y, filt_adj = [], [], []
        newnumarr, newm = [], []

        for (pt_x, pt_y, dpt_x, dpt_y,
            rx, ry, radj,
            slope, label) in results:

            # original loop used u = np.arange(len(red_x))
            #   and then newpt = splev(u, ...)  → pt_x/pt_y are already that
            spline_x.append(pt_x)
            spline_y.append(pt_y)

            smoothskel_x.append(dpt_x)
            smoothskel_y.append(dpt_y)

            filt_x.append(rx)
            filt_y.append(ry)
            filt_adj.append(radj)

            # newm = newder[1]/newder[0] → slope already computed
            newm.append(slope)

            newnumarr.append(label)

        # concatenate – same shapes as original
        spline_x = np.concatenate(spline_x, axis=0)
        spline_y = np.concatenate(spline_y, axis=0)
        smoothskel_x = np.concatenate(smoothskel_x, axis=0)
        smoothskel_y = np.concatenate(smoothskel_y, axis=0)
        filt_x = np.concatenate(filt_x, axis=0)
        filt_y = np.concatenate(filt_y, axis=0)
        filt_adj = np.concatenate(filt_adj, axis=0)
        newnumarr = np.concatenate(newnumarr, axis=0)
        newm = np.concatenate(newm, axis=0)

        # original code transformed the slopes
        newm = -1.0 / newm

        # ------------------------------------------------------------------
        # 7️⃣  Update the object's attributes if requested
        # ------------------------------------------------------------------
        if update:
            self.m_arr = newm
            self.cen_x = spline_x
            self.cen_y = spline_y
            self.adj_dist = filt_adj
            self.filnum = np.array(newnumarr, dtype='int')

        # ------------------------------------------------------------------
        # 8️⃣  Build the dense smooth skeleton image (identical to original)
        # ------------------------------------------------------------------
        smoothskel_x = np.array(smoothskel_x, dtype='int')
        smoothskel_y = np.array(smoothskel_y, dtype='int')
        img_dum = np.zeros_like(self.skel)

        x_ok = (smoothskel_x < img_dum.shape[0]) & (smoothskel_x > 0)
        y_ok = (smoothskel_y < img_dum.shape[1]) & (smoothskel_y > 0)
        keep = x_ok & y_ok

        smoothskel_x = smoothskel_x[keep]
        smoothskel_y = smoothskel_y[keep]

        img_dum[smoothskel_x, smoothskel_y] = 1
        self.smooth_skel = img_dum
        message("Smoothing Done")

        # ------------------------------------------------------------------
        # 9️⃣  Return the same tuple as the original implementation
        # ------------------------------------------------------------------
        return (spline_x, spline_y, filt_x, filt_y, newnumarr, newm)

    def smoothen(self, tol=0.8):
        m_arr_tan = np.arctan(self.m_arr)
        m_arr_grad = np.gradient(np.arctan(self.m_arr))
        m_mask = np.ones(len(m_arr_tan))

        for i in range(0, len(m_arr_tan)):
            if(np.abs(m_arr_grad[i])>tol):
                m_mask[i] = 0

        m_mask = np.array(m_mask, dtype='bool')

        self.m_arr = self.m_arr[m_mask]
        self.cen_x = self.cen_x[m_mask]
        self.cen_y = self.cen_y[m_mask]
        self.adj_dist = self.adj_dist[m_mask]        

    def cut_off_points(self, alpha=None):
        '''
        Cutoff of the radial profile distance. If cutoff distance is given, 
        the number of pixels is computed using the pixel scale. else if pixel 
        number is given, then 
        '''
        # if(alpha isty)
        message("Setting Cuto-off to radial profiles")
        if(alpha is None):
            wcs = WCS(self.header)
            ps = u.pixel_scale((wcs.pixel_scale_matrix[1,1]*u.degree/u.pixel)) 
            theta = (self.meta_info['radial-cutoff']/ self.meta_info['distance'])*u.rad
            self.N_CUT_OFF_PIX = (theta).to(u.pixel , ps)
        else: self.N_CUT_OFF_PIX = alpha
    
        self.cut_off_pix = alpha
        th = np.arctan(self.m_arr)
        # th = np.array([np.arctan(m) if(m<998) else np.pi/2 for m in m_arr])
        self.x_high = np.around(self.cen_x + alpha*np.cos(th))
        self.y_high = np.around(self.cen_y + alpha*np.sin(th))
        self.x_low = np.around(self.cen_x - alpha*np.cos(th))
        self.y_low = np.around(self.cen_y - alpha*np.sin(th))
        message("Done cutoff")

    def get_filament_table(self):
        findxs = np.unique(self.filnum)
        fil_lengths = [len(self.cen_x[self.filnum == fi]) for fi in findxs]
        xl = [self.cen_x[self.filnum==fi][0] for fi in findxs]
        yl = [self.cen_x[self.filnum==fi][-1] for fi in findxs]
        xh = [self.cen_y[self.filnum==fi][0] for fi in findxs]
        yh = [self.cen_y[self.filnum==fi][-1] for fi in findxs]
        df = pd.DataFrame({
            'Index' : findxs , 
            'Length' : fil_lengths , 
            'bbox_x1' : xl ,
            'bbox_y1' : yl ,
            'bbox_x2' : xh ,
            'bbox_y2' : yh ,
        }).set_index('Index')
        return df
    def get_filament_location(self , findex):
        loca = np.asarray([self.cen_x[self.filnum==findex] ,self.cen_y[self.filnum==findex] ])
        return loca
    
    def get_filament(self, findx):
        filnum_arr = np.asarray([p['filnum'] for p in self.prof_dict])
        fil = np.asarray(self.prof_dict)[filnum_arr == findx]
        return fil

    def get_filament_profile(self, filament_index):
        '''
        to get the filament profile of selected filaments
        '''
        img_shape = self.img.shape
        linelist = []
        newxs = []
        newys = []
        filindex = self.filnum == filament_index
        # print(filindex)
        xlow  = self.x_low[filindex][1:-1]
        ylow = self.y_low[filindex][1:-1]
        xhigh = self.x_high[filindex][1:-1]
        yhigh = self.y_high[filindex][1:-1]
        cenx = self.cen_x[filindex][1:-1]
        ceny = self.cen_y[filindex][1:-1]
        # print(xlow)
        prof_dict = []
        dist_list = []  
        colden_list = []
        loc_list = []
        # for xl, yl,xh,yh,xc,yc,fn in zip(self.x_low, self.y_low, self.x_high, self.y_high, self.cen_x, self.cen_y, self.filnum):
        for xl, yl,xh,yh,xc,yc in zip(xlow, ylow, xhigh, yhigh, cenx, ceny):
            if(min(xl, xh)<0 or max(xl, xh)>img_shape[0]-1): continue
            elif(min(yl, yh)<0 or max(yl, yh)>img_shape[1]-1): continue
            else:
                newxs.append([xl,xc,xh])
                newys.append([yl,yc,yh])
                xx, yy = line(int(xl), int(yl), int(xh), int(yh))
                linelist.append([xx, yy])
                one_prof = {
                    'line_x': xx,
                    'line_y': yy,
                    'cen': [xc, yc], 
                    'low': [xl, yl],
                    'high': [xh, yh], 
                    'slope' : (yh-yl / xh-xl) 
                }
                col_val = [self.img[x, y] for x,y in zip(xx, yy)]
                colden_list.append(col_val)
                loc_list.append(np.asarray([[x,y] for x,y in zip(xx,yy)]).T)
                pt = len(xx)

                dmin = np.sqrt((xl-xc)**2 + (yl-yc)**2)
                dmax = np.sqrt((xh-xc)**2 + (yh-yc)**2)
                dist_list.append(np.linspace(-dmin, dmax, pt))
        prof_dict = {
            'colden values' : colden_list , 
            'pixel locations' :  loc_list , 
            'distance_array' : dist_list
        }
        return prof_dict
        
    def create_rad_profile_single_thread(self):
        '''
        creates readial profile of all the filaments globally
        '''
        img_shape = self.img.shape
        linelist = []
        newxs = []
        newys = []

        prof_dict = []

        message("Creating Radial profiles")

        for xl, yl,xh,yh,xc,yc,fn in zip(self.x_low, self.y_low, self.x_high, self.y_high, self.cen_x, self.cen_y, self.filnum):
            if(min(xl, xh)<0 or max(xl, xh)>img_shape[0]-1): continue
            elif(min(yl, yh)<0 or max(yl, yh)>img_shape[1]-1): continue
            else:
                newxs.append([xl,xc,xh])
                newys.append([yl,yc,yh])
                xx, yy = line(int(xl), int(yl), int(xh), int(yh))
                linelist.append([xx, yy])
                one_prof = {
                    'line_x': xx,
                    'line_y': yy,
                    'cen': [xc, yc], 
                    'low': [xl, yl],
                    'high': [xh, yh], 
                    'filnum': fn ,
                    'slope' : np.arctan2((yh-yl) , (xh-xl)) 
                }
                prof_dict.append(one_prof)

        coldens_list = []
        for el,prof in zip(linelist,prof_dict):
            val = [self.img[x, y] for x,y in zip(el[0], el[1])]
            coldens_list.append(np.array(val))
            prof['col_dens'] = np.array(val)

        pix_dist = []
        for el,xvals,yvals,prof in zip(linelist, newxs, newys,prof_dict):
            pt = len(el[0])
            dmin = np.sqrt((xvals[0]-xvals[1])**2 + (yvals[0]-yvals[1])**2)
            dmax = np.sqrt((xvals[2]-xvals[1])**2 + (yvals[2]-yvals[1])**2)
            pix_dist.append(np.linspace(-dmin, dmax, pt))
            prof['pix_dist'] = np.linspace(-dmin, dmax, pt)
            prof['sky_dist'] = self.get_sky_dist(prof['pix_dist'] )

        self.prof_dict = prof_dict
        message("Done creating radial profiles")


    def create_rad_profile(self):
        """Parallel version of the original radial‑profile builder."""
        import multiprocessing as mp
        from skimage.draw import line   # keep the same import used in the original code

        message("Creating Radial profiles")

        img_shape = self.img.shape

        # ------------------------------------------------------------------
        # Pack *only* picklable arguments for each filament
        # ------------------------------------------------------------------
        #  * self.get_sky_dist is a *method* that holds a file handle → cannot be pickled.
        #    Instead we pass a **string key** that identifies which routine to use.
        #    The mapping from key → callable lives in `sutra.profiler.utils.sky_distance_routines`.
        # ------------------------------------------------------------------
        # sky_func_name = getattr(self, "sky_dist_name", "default")   # user can set this attribute
        # sky_kwargs    = getattr(self, "sky_dist_kwargs", {})        # optional extra parameters
        distance = self.meta_info['distance']
        wcs = WCS(self.header)
        pix_scale_matrix = wcs.pixel_scale_matrix[1,1]
        args_list = [
            (xl, yl, xh, yh, xc, yc, fn,
            img_shape, self.img, distance, pix_scale_matrix)
            for xl, yl, xh, yh, xc, yc, fn
            in zip(self.x_low, self.y_low,
                self.x_high, self.y_high,
                self.cen_x, self.cen_y,
                self.filnum)
        ]

        # ------------------------------------------------------------------
        #  Spawn workers
        # ------------------------------------------------------------------
        n_workers = max(1, mp.cpu_count() - 1)
        with mp.Pool(processes=n_workers) as pool:
            raw_results = pool.map(_rad_profile_worker, args_list)

        # ------------------------------------------------------------------
        #  Remove out‑of‑bounds filaments (the worker returns None for those)
        # ------------------------------------------------------------------
        results = [r for r in raw_results if r is not None]

        # ------------------------------------------------------------------
        #  Re‑assemble the original containers for backward compatibility
        # ------------------------------------------------------------------
        if results:                         # guard against the case of *no* valid filaments
            newxs, newys, line_xs, line_ys, prof_dict = zip(*results)

            # The class originally kept three separate lists (`linelist`, `newxs`, `newys`).
            # They are recreated here – they are not used elsewhere but keeping them
            # preserves the exact public API.
            self.linelist = [list(zip(x, y)) for x, y in zip(line_xs, line_ys)]
            self.newxs    = list(newxs)
            self.newys    = list(newys)

            # Store the list of dictionaries exactly as the original method did.
            self.prof_dict = list(prof_dict)
        else:
            # No filament survived the bounds check → initialise empty containers
            self.linelist   = []
            self.newxs      = []
            self.newys      = []
            self.prof_dict  = []

        message("Done creating radial profiles")