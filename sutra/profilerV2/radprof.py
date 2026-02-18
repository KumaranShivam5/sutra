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

import streamlit as st

from scipy.signal import argrelmin , argrelmax
import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import multiprocessing as mp
from typing import List, Tuple

from sutra.logger import message
from matplotlib import pyplot as plt

from astrools.image import make_wcs_good

from sutra.wcs_utils import get_beam_size


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

def _skeleton2tangents(skel, ks=5, stride=5, _n_workers: int | None = None):
    message(f'Finding tangents to the skeleton in multiprocessing mode | Pixel distance {stride} | End padding {ks}')
    n_workers = max(1, (_n_workers if _n_workers is not None else mp.cpu_count() - 1))

    max_i = skel.shape[0] - ks
    slice_edges = np.linspace(0, max_i, n_workers + 1, dtype=int)

    with mp.Pool(processes=n_workers) as pool:
        args = ((skel,
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

    m_arr = np.array(m_arr, dtype=float)
    cen_x = np.array(cen_x, dtype=int)
    cen_y = np.array(cen_y, dtype=int)
    message(f'Tangent finding completed')
    return cen_x, cen_y, m_arr


@st.cache_data
def skeleton2tangents(skel, ks=5, stride=5, _n_workers: int | None = None):
    m_arr = []
    cen_x = []
    cen_y = []
    ks = max(ks, stride)
    for i in range(0, len(skel)-ks, stride):
        for j in range(0, len(skel[0])-ks, stride):
            box = skel[i:i+ks, j:j+ks]
            if(np.sum(box)<3): 
                continue
            inds = np.where(box.flatten())[0]
            x = np.array(inds/ks, dtype='int')
            y = inds - ks*x
            coeff, _, r, _, _ = np.polyfit(x+1,y+1,1, full=True)
            # if(coeff[0]<0.000001): m_arr.append(999)
            if(r<2): m_arr.append(0)
            else: m_arr.append(-1/coeff[0])

            cen_x.append(np.around(np.median(x))+i)
            cen_y.append(np.around(np.median(y))+j)

    m_arr = np.array(m_arr)
    cen_x = np.array(cen_x)
    cen_y = np.array(cen_y)

    return cen_x, cen_y, m_arr

@st.cache_data
def reorder_prof(prof_arr , count_thres = 10 ):
    """
    TODO : must speed-up  O(n^2)
    """
    message("Reordring", 3)

    cen_x, cen_y , m_arr = prof_arr

    count_arr = []
    strt = np.argmin(cen_x)
    count_arr.append(strt)
    ind_arr = np.arange(0, len(cen_x))
    adj_dist =[0.0]
    while len(count_arr)<len(cen_x):
        tempinds = np.setdiff1d(ind_arr, count_arr)
        distarr = np.sqrt((cen_x[tempinds] - cen_x[strt])**2 + (cen_y[tempinds] - cen_y[strt])**2)
        strt = tempinds[np.argmin(distarr)]
        count_arr.append(strt)
        adj_dist.append(distarr[np.argmin(distarr)])

    m_arr = m_arr[count_arr]
    cen_x = cen_x[count_arr]
    cen_y = cen_y[count_arr]
    adj_dist = np.array(adj_dist)

    filnum_arr = []
    filcount = 0
    for i in range(len(m_arr)):
        if(adj_dist[i]>count_thres): filcount+=1
        filnum_arr.append(filcount)

    filnum = np.array(filnum_arr, dtype='int')
    return (cen_x, cen_y,  m_arr , filnum ,adj_dist)
    message("Reordering Done")


def _spline_smooth(loc_arr, update=True, max_knots = None):

    cen_x , cen_y , filnum ,  adj_dist  = loc_arr
    # if max_knots is not None
    max_knots = int(max_knots)
    # max_knots = 12
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
    for i in range(0, filnum[-1]+1):
        red_x = cen_x[filnum == i][1:-1]
        red_y = cen_y[filnum == i][1:-1]
        red_adj = adj_dist[filnum == i][1:-1]
        if(len(red_x)<5): 
            minus+=1
            continue

        u = np.arange(0, len(red_x))
        mkf = int(np.around(len(red_x)/max_knots))

        tck, uf = splprep([red_x, red_y], k = 3, u=u, nest=mkf)
        # tck, uf = splprep([red_x, red_y], k = 3, u = u, s = 0.5)
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

    filnum = np.array(newnumarr, dtype='int')

    return spline_x , spline_y , newm , filnum , filt_adj
    if(update==True):
        m_arr = newm
        cen_x = spline_x
        cen_y = spline_y
        adj_dist = filt_adj
        filnum = np.array(newnumarr, dtype='int')

    # smoothskel_x = np.array(smoothskel_x, dtype='int')
    # smoothskel_y = np.array(smoothskel_y, dtype='int')
    # img_dum = np.zeros_like(self.skel)
    # xind = (smoothskel_x < img_dum.shape[0])*(smoothskel_x > 0 )
    # yind = (smoothskel_y < img_dum.shape[1])*(smoothskel_y > 0)
    # ind = [xi and yi for xi,yi in zip(xind , yind)]

    # smoothskel_x = smoothskel_x[ind]
    # smoothskel_y = smoothskel_y[ind]
    # # return smoothskel_x , smoothskel_y
    # img_dum[smoothskel_x, smoothskel_y] = 1
    # self.smooth_skel = img_dum

    message('Smooth done')


    return(spline_x, spline_y, filt_x, filt_y, newnumarr, newm)


from sutra.wcs_utils import get_sky_dist , get_pixel_to_arcsec
from sutra.profilerV2.beamProps import profGroup
from sutra.profilerV2.reconnect import get_connected_struct_from_mst
from tqdm.notebook import tqdm
import pandas as pd

from sutra.profilerV2.prob2skl import filter_background

class RadProf:
    def __init__(self, img, skeleton , meta_info = None):
        self.img = img.data
        self.header = img.header
        self.hdu = img
        self.skel = skeleton
        self.meta_info = meta_info
        self.hpbw , self.pixel_size = get_beam_size(self.header , self.meta_info['beam'])
        self.bkg_mask , self.bkg_th = filter_background(img.data)
    
    def _get_sky_dist(self, pixdist):
        # wcs = WCS(self.header)
        tdist = get_sky_dist(self.header, pixdist , self.meta_info['distance'])
        return tdist
    
    # def get_sky_dist(self, pixdist):

    # def set_cut_off(self , cut_off_size = None):

    def pc_to_pixel(self , pcdist):
        #conversion factor
        cf = 1 / self._get_sky_dist(pixdist = [1])[0]
        # return cf
        return np.asarray(pcdist)*cf

    def tangents(self, ks=5, stride=3, _n_workers: int | None = None):
        x,y,m = skeleton2tangents(self.skel, ks=ks, stride=stride, _n_workers = None)
        self.m_arr = np.array(m, dtype=float)
        self.cen_x = np.array(x, dtype=int)
        self.cen_y = np.array(y, dtype=int)
        # message(f'Tangent finding completed')
    
    def reorder(self, count_thres = 10 ):
        x ,y,m,f ,a = reorder_prof((self.cen_x , self.cen_y, self.m_arr ))
        self.cen_x , self.cen_y, self.m_arr , self.filnum , self.adj_dist = x,y,m,f, a 
    
    def spline_smooth(self):
        x,y,m , fn,ad = _spline_smooth(loc_arr=(self.cen_x , self.cen_y, self.filnum , self.adj_dist), max_knots=self.hpbw.value)
        self.m_arr = m
        self.cen_x = x
        self.cen_y = y
        self.adj_dist = ad
        self.filnum = fn

    def cut_off_points(self, alpha_pc=None):
        '''
        Cutoff of the radial profile distance. If cutoff distance is given, 
        the number of pixels is computed using the pixel scale. else if pixel 
        number is given, then 
        '''
        # if(alpha isty)
        message("Setting Cuto-off to radial profiles")
        if(alpha_pc is None):
            alpha = int(self.pc_to_pixel(0.6)/2)
            # print()
            message(f"Setting profile radial cutoff distance to 0.6 pc ({alpha} pix)", 2)
        else:
            alpha = int(self.pc_to_pixel(alpha_pc)/2)
            message(f"Setting profile radial cutoff distance to {alpha_pc} pc ({alpha} pix)", 2)
        self.N_CUT_OFF_PIX = alpha
        # message(alpha, 2)
        self.cut_off_pix = alpha
        th = np.arctan(self.m_arr)
        # th = np.array([np.arctan(m) if(m<998) else np.pi/2 for m in m_arr])
        self.x_high = np.around(self.cen_x + alpha*np.cos(th))
        self.y_high = np.around(self.cen_y + alpha*np.sin(th))
        self.x_low = np.around(self.cen_x - alpha*np.cos(th))
        self.y_low = np.around(self.cen_y - alpha*np.sin(th))
        message("Done cutoff")

    def create_rad_profile_single_thread(self):
        '''
        creates readial profile of all the filaments globally
        # TODO : this function is bottleneck
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
            prof['sky_dist'] = self._get_sky_dist(prof['pix_dist'] )

        self.prof_dict = prof_dict
        message("Done creating radial profiles")

    def get_filament(self, findx):
        filnum_arr = np.asarray([p['filnum'] for p in self.prof_dict])
        fil = np.asarray(self.prof_dict)[filnum_arr == findx]
        return fil
    
    def get_filament_index(self):
        return [i['filnum'] for i in self.prof_dict]
    
    def group_profiles(self,  stride = 1, NORM = 1e21): 
        N_profiles = len(self.prof_dict)
        N_filaments = np.unique(self.get_filament_index())
        # findx  = 1
        beam_filament_index = np.asarray([])
        beam_group_arr = np.asarray([])
        message(f"Grouping filaments based on {stride}* beam size", 3)
        for findx in tqdm(N_filaments):
            # message(f"{findx}", 2)
            fil = self.get_filament(findx)
            if(len(fil)<2) : continue # cases when only one profile is abailable for the filament (at image boundary)
            beam = self.hpbw.value

            cent = np.asarray([p['cen'] for p in fil])
            # compute distances between centers. do cumulative sum. remainder it with the size of chunk required
            g_sum = np.sqrt(np.square(cent[1:]-cent[:-1]).sum(axis=1)).cumsum() 
            g = g_sum % np.around((beam*stride)) # last bracket is extremely improtant
            #np.around is important to group end points
            # Compute location of changes, where the size difference reaches hpbw*stride
            mask = g[:-1]>g[1:]
            # get index of those locations
            indx = np.arange(0, len(g)-1)
            chunk_locs = indx[mask]
            #starting from zero
            # if chunk_locs[0]!=0: # if starting position is already zero
            chunk_locs = np.insert(chunk_locs ,  0 , 0)
            if len(np.unique(chunk_locs)) < len(chunk_locs) : continue # sometime 0 gets repeated
            # print(len(mask))
            pg_arr = [profGroup(fil[c1:c2], beam = beam, dist_modifier=self._get_sky_dist) for c1,c2 in zip(chunk_locs[:-1],chunk_locs[1:])]
            pg_arr.append(profGroup(fil[chunk_locs[-1]:], beam = beam, dist_modifier=self._get_sky_dist))
            beam_group_arr = np.append(beam_group_arr, pg_arr)
            # track which filament the beam belongs to
            beam_filament_index = np.append(beam_filament_index, np.ones(len(pg_arr))*findx)
        # beam_group_arr = beam_group_arr
        beam_filament_index = beam_filament_index
        beam_cen =  np.asarray([[bg.gc[0], bg.gc[1]] for bg in beam_group_arr])
        xb, yb = beam_cen.T 
        m = [bg.m for bg in beam_group_arr]
        R = self.N_CUT_OFF_PIX / 2
        xl , xh = xb  - R * np.cos(np.arctan(m))  , xb  + R * np.cos(np.arctan(m)) 
        yl , yh = yb  - R * np.sin(np.arctan(m))  , yb  + R * np.sin(np.arctan(m)) 

        beam_arcsec = get_pixel_to_arcsec(self.header , self.hpbw.value)

        message(f"Reconnecting beams using MST | Distance tolerance {2*beam_arcsec:.2f} arcsec ({self.hpbw.value:.0f} pix) ", 3)
        # rejoin skeleton using MST
        new_beam_filament_index , _ , _ = get_connected_struct_from_mst(
            xb, yb, beam_filament_index , 
            distance_tol = 2*self.hpbw.value)
        new_beam_filament_index = np.copy(beam_filament_index)
        


        self.beam_dict = {
            "fil_index" : new_beam_filament_index , 
            "cen" : np.asarray(beam_cen) , 
            "slope" : np.asarray(m) , 
            "beam_elements" : beam_group_arr,
            "low" : np.asarray([xl,yl]) , 
            "high" : np.asarray([xh,yh]) , 
            # pro
        }


    def get_all_beam_props(self, refresh = False):
        if self.beam_dict is None :
            raise ValueError('Run group profiles first')
        
        message("Computing Width and LMD of each beams")
        larr , lmdarr , Rbg_r , Rbg_l = [] , [] , [] , []
        i = 0
        for p in tqdm(self.beam_dict['beam_elements'], leave=False, desc = 'Fil-Properties'):
            p = p.get_props(refresh)
            larr.append(p['Length'])
            lmdarr.append(p['LMD'])
            Rbg_r.append(p['RF']['R_bg'])
            Rbg_l.append(p['LF']['R_bg'])

        props = {}
        props['beamIndex'] = np.arange(len(self.beam_dict['beam_elements']))
        


        message("Fitting plummer profiles to wach beam")
        red_chi_arr , Nfil , Rflat , Pindx , Nbg = [] , [] , [] , [] , []
        x,y = self.beam_dict['cen'].T
        fid = np.asarray(self.beam_dict['fil_index'], dtype='int')
        props['filID'] = fid
        bid = np.empty_like(fid)
        for u in np.unique(fid): bid[fid==u] = np.arange((fid==u).sum())
        props['beamID'] = bid
        props['X'] = x 
        props['Y'] = y
        sk = WCS(self.header).pixel_to_world(props['Y'] , props['X'])
        props['gal_l'],props['gal_b'] = sk.galactic.l.value , sk.galactic.b.value

        props['Length'] = larr
        props['Slope'] = np.arctan(self.beam_dict['slope'])
        props['Rbg_r'] = Rbg_r
        props['Rbg_l'] = Rbg_l
        # props['Rbg_s'] = Rbg_l+Rbg_r
        props['LMD'] = lmdarr 

        pi = []
        for p in tqdm(self.beam_dict['beam_elements'], leave='False', desc = "Plummer fit"):
            pi.append(p.get_plummer_fit(refresh))

        plummerProps = pd.DataFrame(pi)

        beamProps = pd.DataFrame(props)
        beamProps.set_index('beamIndex', inplace=True)

        plummerProps.index.name = 'beamIndex'
        beamProps = pd.merge(
            beamProps , plummerProps , 
            left_index=True, right_index=True)
        beamProps.loc[:,'red-chi-avg'] = (beamProps['red-chi-L'] + beamProps['red-chi-R'])/2
        beamProps['Nbg-avg'] = np.maximum((beamProps['Nbg_L']+beamProps['Nbg_R'])/2 , self.bkg_th/1e21)
        beamProps['FC'] =   (beamProps['Nfil']-beamProps['Nbg-avg']) /  beamProps['Nbg-avg']
        self.beamProps = beamProps.reset_index().set_index(['filID', 'beamID', 'beamIndex'])
        return self.beamProps
    

    def plot_filament(self, findx , sizeby = "Pi-L" , colorby = 'LMD' , ax = None, show_beamid = False, sizescale = 1, red_chi_filter = 3, contrast_filter = 0.3 , **cdplot_kwargs):


        fprops = self.beamProps.loc[findx]
        beam_index = fprops.index.get_level_values(1)

        fprops.loc[:,'flag'] = (fprops['FC'] > contrast_filter) & (fprops['red-chi-avg']<red_chi_filter)
        # display(fprops)
        if ax is None:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111, projection = WCS(self.header))
        # ax = plt.subplot(111, projection=WCS(self.header) )

        xmin, xmax = int(np.floor(fprops['X'].min())),int( np.ceil(fprops['X'].max()))
        ymin, ymax = int(np.floor(fprops['Y'].min())),int(np.ceil(fprops['Y'].max()))


        selected = np.asarray(self.img)[xmin:xmax, ymin:ymax]
        sel_cd = np.asarray(self.img)[xmin:xmax, ymin:ymax]
        vmin , vmax = np.percentile(sel_cd, q = [2,90])

        sel_skl = np.asarray(self.skel)[xmin:xmax, ymin:ymax]
        
        ax.imshow(self.img , cmap = 'Greys_r', norm = 'log', vmin = vmin, vmax = vmax,)
        ax.contour(self.skel, colors = 'r', linewidths = 0.2, alpha = 0.2, zorder  =1, )
        ax.plot([],[], c = 'r', label='Skeleton', lw  = 1)
        to_plot = fprops[fprops['flag']]
        sc = ax.scatter(to_plot['Y'] , to_plot['X'], c = to_plot[colorby], s = sizescale*to_plot[sizeby] , cmap = 'Reds', zorder = 3, vmin=0, vmax = 3, label='size')
        xl, yl = self.beam_dict['low'][:,beam_index]
        xh, yh = self.beam_dict['high'][:,beam_index]

        ax.plot([yl,yh], [xl,xh] , c = 'teal', lw = 2, alpha = 0.5, zorder  = 2)
        # cbar = plt.colorbar(sc, shrink=0.8, )
        arr1 = ~fprops['flag'].to_numpy()
        arr2 = fprops.index.get_level_values(1).to_numpy()
        bi = split_array(arr1, arr2)
        for b in bi:
            ax.plot(self.beamProps.loc[:,:,b]['Y'], self.beamProps.loc[:,:,b]['X'], lw = 5, c = 'k')
        if show_beamid:
            b = fprops.index.get_level_values(1)
            x = fprops['X']
            y = fprops['Y']
            for xi,yi,bi in zip(x,y,b):
                ax.text(yi+2,xi+2, bi, c = 'b', fontsize=10)
        ax = make_wcs_good(ax)
        # ax.legend(loc=2, fontsize = 10)
        # cbar0 = fig.colorbar(im0, ax=ax[0], orientation='horizontal',
        #             location='top', pad=0.1, shrink=0.9)
        ax.set_ylim(xmin-10, xmax+10)
        ax.set_xlim(ymin-10, ymax+10)
        return ax
    
    def plot_props(self, sizeby = "Pi-L" , colorby = 'LMD' , ax = None, show_filid = False, sizescale = 1, red_chi_filter = 3, **cdplot_kwargs):
        if ax is None:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111, projection = WCS(self.header))
        vmin , vmax = np.percentile(self.img, q = [2,98])
        
        ax.imshow(self.img , cmap = 'Greys_r', norm = 'log', vmin = vmin, vmax = vmax, **cdplot_kwargs)

        ax.contour(self.skel, colors = 'r', linewidths = 0.1, alpha = 1, zorder  =1)
        props = self.beamProps
        to_plot = props[props['red-chi']<=red_chi_filter]
        print(len(to_plot))

        # cmap_dict = {}

        if colorby=='red-chi':
            sc = ax.scatter(to_plot['Y'] , to_plot['X'], c = to_plot[colorby], s = sizescale*to_plot[sizeby] , cmap = 'RdYlGn_r', zorder = 3 , vmin = 1, vmax = 3)
        elif colorby=='Nfil':
            sc = ax.scatter(to_plot['Y'] , to_plot['X'], c = 1e21*to_plot['Nfil'], s = sizescale*to_plot[sizeby], cmap = 'Reds',  norm='log',)
        else: 
            sc = ax.scatter(to_plot['Y'] , to_plot['X'], c = to_plot[colorby], s = sizescale*to_plot[sizeby] , cmap = 'Reds')
        filnum = self.beam_dict['fil_index']
        xl, yl = self.beam_dict['low']
        xh, yh = self.beam_dict['high']
        ax.plot([yl,yh], [xl,xh] , c = 'teal', lw = 0.5, alpha = 0.5, zorder  = 2)
        ax = make_wcs_good(ax)
        ax.set_xlim(0,self.img.shape[0])
        ax.set_ylim(0,self.img.shape[1])
        if show_filid:
            for indx in np.unique(filnum):
                fil_label_loc = np.where(filnum==indx)[0][0]
                fil_label_x , fil_label_y = props['Y'].to_numpy()[fil_label_loc] , props['X'].to_numpy()[fil_label_loc]
                fil_label_text = f'{int(indx)}'
                fil_label_x , fil_label_y
                ax.text(fil_label_x, fil_label_y, fil_label_text, zorder = 10 , c = 'teal', fontsize = 10, fontweight='bold')
        # cbar1 = plt.colorbar(sc, ax=ax, shrink=1)
        # cbar1.set_label(colorby)
        
        plt.tight_layout()
        return ax , sc


def split_array(arr1, arr2):
  """Splits arr2 based on True values in arr1."""
  indices = np.where(arr1)[0]
  result = []
  start = 0
  for i in indices:
    result.append(arr2[start:i])
    start = i + 1
  result.append(arr2[start:])
  return result
