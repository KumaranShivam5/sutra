
from scipy.signal import argrelmin , argrelmax
import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

NORM =  1e21

from .profiling import RadProf
   
def plummer_fn(x , amplitude , R , p , bg):
    '''
    Function to fit plummer profile
    params::
        x : distance array
        amplitude : amplitude
        R : R_flat 
        p : plummer index 
        bg : background
    '''
    nom =  amplitude 
    denom = (1 + (x/R)**2)**((p-1)/2) 
    prof_val = nom / denom + bg
    return prof_val

def gaussian_fn(x , amplitude , sigma , bg):
    return amplitude* np.exp(-x**2 / (2*sigma**2)) +bg

def get_dist(pix_dist , wcs , dist=None):
    sk_dist = pix_dist*wcs.proj_plane_pixel_scales()[0].to(u.rad).value*dist*u.pc
    sk_dist = [si.value for si in sk_dist]
    return sk_dist

from astropy.convolution import Gaussian1DKernel, convolve , convolve_fft

class filProfile():
    '''
    This class contains filament profile (variation of values along one pixel wide line perpendicular to the filament tangent. Probable value of filament is column density, or can be something else depending on the map on which filament properties are being calculated) and the corresponding operations
    '''
    def __init__(self, dist, prof): #distance array and profile value
        self.dist = dist
        self.prof = prof
        self.cent = None
        self.smooth_prof = None

    def convolve_prof(self, beam):
        gk = Gaussian1DKernel(stddev=beam / 2.355)
        colden_smooth = convolve(self.prof , gk, boundary='extend')
        self.smooth_prof = colden_smooth

    def get_center(self):
        if(self.convolve_prof):
            mxindx = np.nanargmax(self.smooth_prof)
        else:
            mxindx = np.nanargmax(self.dist)
        self.cent = self.dist[mxindx]

    def fold(self):
        right = dist

    def plot(self, which='full', ax = None):
        '''
        Plot filament
        '''
        if(ax is None):
            plt.figure(figsize = (5,3))
            axi = plt.subplot(111)
        else: axi=ax
        if(self.smooth_prof is not None):
            axi.plot(self.dist, self.smooth_prof, c = 'k', label='convolved')
        axi.plot(self.dist, self.prof, c = 'crimson', label='profile')
        if(self.cent is None): self.get_center()
        axi.axvline(self.cent)
        axi.set_xlabel('Distance from filament crest')
        axi.set_ylabel('Profile value')
        axi.legend()
        if(ax is None):
            plt.show()
        else: return axi


def group_profiles(profs, side='both'):
    '''
    This function groups N filament profiles (pix-dist on x axis and column density on y-axis). Each profiles are individually convolved with 
    '''







def fit_plummer_profile(pol : RadProf, FIL : int , side=1):
    '''
    Function to fil Plummer profile to the filament
    We first fold each radial profiles at the maxima and then
    seperate it into right and left
    
    Parameters
    ----------
    pol : RadProf

    FIL : int
        Index of the filament to be fitted

    Returns
    -------
    dict : 
        'fit_params' : param[0] ,
        'param error' : param[1] ,
        'distance' : np.asarray(dist_arr) , 
        'Obs CD' : mn , 
        'masked Obs CD' : mn[mask] , 
        'Model CD' : model_res ,
        'median err' : mad , 
        'r-score' : rscore , 
        'R_bg' : r_bg, 
        'hr' : hr # half radius
    '''
    exprof = pol.get_filament_profile(FIL)
    nprof = len(exprof['colden values'])
    N_CUT_OFF_PIX =  pol.cut_off_pix+2
    prof_arr_right  = np.ones((nprof , N_CUT_OFF_PIX*2))
    prof_arr_left  = np.ones((nprof , N_CUT_OFF_PIX*2))

    gk = Gaussian1DKernel(stddev=pol.hpbw.value / 2.355)
    for indx in range(nprof):
        colden , dist = np.asarray(exprof['colden values'][indx]) , exprof['distance_array'][indx]
        colden[colden<pol.global_threshold] = np.nan
        mxindx = np.nanargmax(convolve(colden , gk))
        # print(mxindx)
        right , left = colden[:mxindx][::-1] , colden[mxindx:] 
        # print(left)
        lmin = argrelmin(left , order=2)[0]
        rmin = argrelmin(right , order =2)[0]
        prof_arr_left[indx][:len(left)] = left
        prof_arr_right[indx][:len(right)] = right

    L_steps = nprof
    print(FIL, L_steps)
    N_steps = math.ceil(nprof / L_steps)
    indx = 0
    pix_dist = np.arange(N_CUT_OFF_PIX*2)
    wcs = WCS(pol.header)
    if(pol.meta_info['distance']):
        dist_arr = get_dist(pix_dist , wcs , pol.meta_info['distance'])
    else: raise ValueError('provide distance as meta_info dictionary')

    if(side==1): current_prof = prof_arr_left
    elif(side==0) : current_prof = prof_arr_right
    else: raise ValueError('Side must be 1 (left) or 2 (right)')

    current_prof[current_prof == 1] = np.nan
    to_plot = current_prof[int(L_steps)*indx:int(L_steps)*(indx+1)]
    # print(to_plot)
    # print(to_plot)
    to_plot = to_plot / NORM
    mn = np.nanmedian(to_plot , axis = 0)
    mad = np.nanmedian(np.abs(to_plot - np.nanmedian(to_plot, axis=0)), axis=0)
    ysig = np.nanstd(to_plot , axis=0)
    mask = np.isfinite(mn)
    # print(mask)
    xdata  =  np.asarray(dist_arr)[mask]
    ydata = mn[mask]
    ysig = ysig[mask]
    dist_arr = np.asarray(dist_arr)
    dist_arr = dist_arr[mask]

    # param = curve_fit(plummer_fn , xdata , ydata , p0 = [np.mean(ydata) , 0.1 , 2 , np.mean(ydata)] , 
    #                 bounds=([min(ydata) ,  0.001 , 1 , min(ydata) ] , [max(ydata)*(2) ,  0.5 , 5 , max(ydata)]))
    # ydata = ydata
    # Here we are trying to find out the first point where slope of the profile becomes 0
    # The logrithmic slope is (dNh2/dr)*(r/Nh2). Using the beamsize (HPBW ~ 36.4 arcses)
    # for low res CD map, the slope is deconvolved. R_out is considered the first point
    # at which the logirithmic slope crosses zero
    # print(len(ydata))
    try:
        dn = np.gradient(ydata)
        dr = np.gradient(xdata)
        slope = dn/dr * (xdata/ydata) # slope
        gk = Gaussian1DKernel(stddev=pol.hpbw.value / 2.355) # gaussian kernel
        convolve_index = (convolve(slope , gk)>0)
        # print(convolve_index)
        if(sum(convolve_index)> 0 and ~convolve_index[0]): # if profile is not good (profile is increasing from 0 itself)
            r_out_index = np.arange(len(xdata))[convolve_index][0] # rout is the first point where it crosses zero
        else : r_out_index = len(xdata)-10
        rout = xdata[r_out_index]
        # print(r_out_index, len(xdata))
        # print(np.nanmin(ydata) , np.nanmax(ydata) , np.mean(ydata))
        param = curve_fit(plummer_fn , xdata[:r_out_index] , ydata[:r_out_index] , p0 = [np.mean(ydata) , 0.2 , 2 , 2*np.mean(ydata)] , 
                        bounds=([np.nanmin(ydata),  0.001 , 0.5 , np.nanmin(ydata) ] , [np.nanmax(ydata) ,  2 , 5 , 2*np.nanmax(ydata)]) , maxfev=5000, method='trf')
        
        a , r , p , bg = param[0]
        #print(ydata.shape , dist_arr.shape)
        model_res = plummer_fn(np.asarray(dist_arr) , amplitude=a , R = r, p = p , bg = bg)
        # model_res = convolve(model_res , gk)
    
        # gparam = curve_fit(gaussian_fn , xdata[:20] , ydata[:20] , p0 = [np.max(ydata) , 0.05 , 0.2*np.std(ydata)] , bounds = ([np.min(ydata)*0.5, 0 , np.std(ydata)*0.1] , [np.max(ydata), 1 , np.std(ydata)]))
        # ga, gr, gbg = gparam[0]
        # gmodel_res = gaussian_fn(np.asarray(dist_arr) , amplitude=ga , sigma = gr,  bg = gbg)
        model_fil = model_res -bg
        hr_index = np.arange(len(model_res))[model_fil > np.max(model_fil)/2]
        # hr_index = np.where(model_fil > np.max(model_fil)/2)[0]
        hd = 2*dist_arr[hr_index[-1]]
        # return hd
        # print(hr_index, len(model_res))
        # sds
    
        num = (ydata[:r_out_index] - model_res[:r_out_index])**2 
        denom = (ydata[:r_out_index] -  np.mean(ydata[:r_out_index]))**2
        rscore = 1 - np.sum(num) / np.sum(denom)
        # print(rscore)
        try:
            r_bg = np.asarray(dist_arr)[np.where(model_res-bg <= 0.1)[0][0]]
        except : r_bg = None
        rdict = {
            'fit_params' : param[0] ,
            'param error' : param[1] ,
            'distance' : np.asarray(dist_arr) , 
            'Obs CD' : mn , 
            'Model CD' : model_res ,
            'masked Obs CD' : mn[mask] , 
            'median err' : mad , 
            'r-score' : rscore , 
            'R_bg' : r_bg , 
            'R_out' : rout, 
            'hd' : hd
        }
    except:
        rdict = {
            'fit_params' : [np.nan]*4 ,
            'param error' : [np.nan]*4 ,
            'distance' : np.asarray(dist_arr) , 
            'Obs CD' : mn , 
            'Model CD' : np.nan ,
            'masked Obs CD' : mn[mask] , 
            'median err' : mad , 
            'r-score' : np.nan , 
            'R_bg' : np.nan , 
            'R_out' : np.nan, 
            'hd' : np.nan
        }
    return rdict

def plot_plummer_fit(pol, FIL, side = 1 , ax1 = None):

    # param , dist_arr , mn , mad , model_res , _ = fit_plummer_profile(pol , FIL)
    fit_res = fit_plummer_profile(pol, FIL, side = side)
    param , dist_arr , mn , mad , model_res , r_bg , r_out = fit_res['fit_params'] ,fit_res['distance'], fit_res['Obs CD'], fit_res['median err'], fit_res['Model CD'] , fit_res['R_bg']  , fit_res['R_out'] 
    bg = param[3]
    fil_emission = mn
    mask = np.isfinite(mn)
    mn , fil_emission , mad = mn[mask] , fil_emission[mask] , mad[mask]
    # dist_arr = dist_arr[mask]
    # print(dist_arr.shape , mn.shape , model_res.shape)
    # fig = plt.figure(layout="constrained" , figsize = (6,4))
    # gs = GridSpec(4, 1, figure=fig)
    # ax1 = fig.add_subplot(gs[:4,0])

    ax1.plot(dist_arr, fil_emission , c = 'black', linewidth = 2, marker = '.', zorder = 2, label = 'Obs. Radial Profile')

    ax1.errorbar(dist_arr , y = fil_emission, yerr = mad, color='gray' , alpha = 1 , zorder = 1 , elinewidth=1,)

    ax1.plot(dist_arr , (model_res), linewidth = 2, c = 'crimson' , label  = 'Plummer Profile', ls = '-')
    # ax1.plot(dist_arr , (gmodel_res) , linewidth = 2, c = 'blue' , label  = 'gaussian Profile', ls = '--')
    ax1.axhline(bg  , ls = ':' , c = 'blue' , label=r'$N^{bg}_{H_2}$')
    # if(len(r_ou))
    if(r_out is not None):
        if(r_out > 0 and r_out < np.max(dist_arr)):
            ax1.axvline(r_out , ls = '--' , c = 'gray', label = r'$R_{bg}$ = '+f'{r_out:.3f} pc')
            # ax1.text(r_out*(1.1), np.nanmin((model_res-bg))+1 , r'$R_{bg}$'+f'={r_out:.3f} pc', rotation=90, va='bottom' , ha = 'left')

    ax1.legend(loc = 1)
    ax1.set_xlabel('Distance from filament spine (pc)')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$N_{H_2}$ ($10^{21}cm^{-2}$)')
    ax1.tick_params(axis="both", direction='in', length=2, which='both')
    # plt.savefig(f'NOTES/images/taurus-filament-radial-profile-fit-{FIL}-left.png' , dpi=300 , bbox_inches = 'tight')
    ax1.set_xlim()
    return ax1