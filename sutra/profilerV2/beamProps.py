

from scipy.integrate import simpson , trapezoid
# from sutra.profiler .plummer import plummer_fit_profile

from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, convolve
import streamlit as st

from sutra.profilerV2.plummer import plummer_fit_profile

import numpy as np
from sutra.logger import message

from matplotlib import pyplot as plt

# def nanmedian_safe(arr, axis=None):
#   """
#   Calculates the nanmedian, handling all-NaN slices without warning.
#   Returns NaN for slices that are all NaN.
#   """
#   arr = np.asarray(arr)
#   if axis is None:
#     if np.all(np.isnan(arr)):
#       return np.nan
#     return np.nanmedian(arr)
#   else:
#     # Handle cases where entire axis is NaN *before* nanmedian
#     if np.all(np.isnan(arr.flatten()), axis=axis):
#       result = np.full(np.shape(arr)[axis], np.nan) #return array of nan
#       return result
#     else:
#       result = np.nanmedian(arr, axis=axis)
#       return result

class filProfile():
    '''
    This class contains filament profile (variation of values along one pixel wide line perpendicular to the filament tangent. Probable value of filament is column density, or can be something else depending on the map on which filament properties are being calculated) and the corresponding operations. 
    This class is meant for handling individual profile
    '''
    def __init__(self, dist, prof, beam = 3, NORM = 1e21): #distance array and profile value
        self.dist = dist
        self.prof = prof
        if(NORM is not None):
            self.prof = self.prof/NORM
        self.reprofile()
        self.cent = None
        self.beam = beam
        self.smooth_prof = None
    
    def reprofile(self):
        '''
        Convert the dist array such that each the difference between individual array elements in the dist_arry is 1 pixels. 
        Currently the difference between each array index on dist_array is not 1 pixels (compare the cases of radial profile direction being parallel vs inclined at 45 deg)
        For doing statistical study on N prifiles we ned to reindex each profile such that each index correspond to uniform distances
        '''
        x , y = self.dist , self.prof
        xi = np.arange(np.min(x), np.max(x), 1)
        self.dist = xi # TODO : convert this pixel array to actual parsec distance array
        self.prof = interp1d(x,y, kind='linear')(xi)

    def convolve_prof(self, beam = None):
        '''
        Before finding out the center of profile, we need to make it smooth, as some noisy data can result in wrong center identification.
        '''
        if (beam is None): beam = self.beam
        if self.smooth_prof is None:
            gk = Gaussian1DKernel(stddev=beam / 2.355) # gaussian kernel using beam size given by the HPBW in radProf class
            colden_smooth = convolve(self.prof , gk, boundary='extend')
            self.smooth_prof = colden_smooth

    def get_center(self):
        '''
        Maxima of convolved profile. The profile can overlap with nearby profiles. The search for maxima should not go beyond certain distance
        '''
        if(self.smooth_prof is None): self.convolve_prof(self.beam)
        from scipy.signal import argrelmin , argrelmax
        maximas = argrelmax(self.smooth_prof)
        lnmx = 2 # lentgh of array holding maximas
        dist = 12 # TODO : replace this distance based on actual pixel beam size
        close_to_zero = np.where(abs(self.dist)<dist)
        mxindx = np.intersect1d(maximas, close_to_zero)
        lnmx = len(mxindx)
        # this situation occurs when two filaments almost overlap and can not be resolved within beam size. in that case all the properties extracted will be wrong and hence we set the profile to nan
        if(lnmx==0) : 
            # what if there is no maxima befolre folding while finding out the center, that does not correspond to a valid filament
            # if you see NAN in individual line profiles, this is the reason
            mxindx = np.where(abs(self.dist)<0.5)[0][0]
            self.prof = [np.nan]*len(self.prof)
        elif(lnmx>1):
            # in this situation, within the specified distance from zero, we have two maximas, we choose the major maxima of the two
            mxindx = mxindx[np.argmin(np.abs(self.dist[mxindx]))]
            # print(np.argmax(self.prof[mxindx]))
        else:
            mxindx = mxindx[0]
        self.cent = self.dist[mxindx]
        self.cent_indx = mxindx

    def fold(self):
        '''
        Once we have identified the index of maxima point, the radial profile is divided into let and right profile and dist array is re-indexed such that each profiles starts form zero distance
        '''
        self.dist_right = self.dist[self.cent_indx:] # select right side of profile
        self.dist_right = np.abs(self.dist_right - self.dist_right[0]) # make sure the starting point is zero and convert to distances using abs
        self.prof_right = self.prof[self.cent_indx:]
        self.dist_left = self.dist[:self.cent_indx+1][::-1]
        self.dist_left = np.abs(self.dist_left-self.dist_left[0])
        # self.dist_left = self.dist_left - np.min(self.dist_left)
        self.prof_left = self.prof[:self.cent_indx+1][::-1]
        if(self.smooth_prof is None): self.convolve_prof()
        self.sm_prof_right = self.smooth_prof[self.cent_indx:]
        self.sm_prof_left = self.smooth_prof[:self.cent_indx+1][::-1]

    def plot(self, which='full', ax = None, ylabel='Profile Value'):
        '''
        Plot filament
        '''
        if(ax is None):
            plt.figure(figsize = (5,3))
            axi = plt.subplot(111)
        else: axi=ax
        if(which=='full'):
            if(self.smooth_prof is not None):
                axi.plot(self.dist, self.smooth_prof, c = 'teal', label='convolved')
            axi.plot(self.dist, self.prof, c = 'teal', label='Observed', lw=1 , ls='--')
            if(self.cent is None): self.get_center()
            axi.axvline(self.cent, lw=1, ls='--', c = 'blue')
            axi.legend(loc=3, fontsize = 10)

        elif(which=='folded'):
            self.fold()
            axi.plot(self.dist_right, self.sm_prof_right, label='Right Profile', c = 'k')
            axi.plot(self.dist_right, self.prof_right,  c = 'k', lw=1 , ls='--')
            axi.plot(self.dist_left, self.sm_prof_left, label='Left Profile', c='crimson')
            axi.plot(self.dist_left, self.prof_left,  c='crimson', lw=1 , ls='--')
            # axi.set_xscale('log')
            axi.legend(loc=3, fontsize = 10)
        
        axi.set_xlabel('r (Pixels)')
        axi.set_ylabel(ylabel)
        # axi.set_xlabel('Distance from filament spine (pc)')
        # axi.set_xscale('log')
        axi.set_ylabel(r'$N(H_2)$ ($10^{21}cm^{-2}$)')
        axi.tick_params(axis="both", direction='in', length=2, which='both')
        if(ax is None):
            plt.show()
        else: return axi


class fitProf():
    '''
    This class handls one profile (ideally the median profile) and its properties
    '''
    def __init__(self, dist,prof,yerr,ystd, beam = 3):
        self.dist = dist
        self.prof = prof
        self.err = yerr
        self.ystd = ystd
        self.beam = beam
        self.get_r_bg()
        self.fil_props = None
        # print(self.beam)
    
    def get_r_bg(self): # Filament radius where it meets background
        xdata, ydata = self.dist , self.prof
        dn = np.gradient(ydata)
        dr = np.gradient(xdata)
        # print(ydata)
        slope = dn/dr * (xdata/ydata) # slope
        gk = Gaussian1DKernel(stddev = self.beam/2/ 2.355) # gaussian kernel
        self.slope = slope
        self.smooth_slope = convolve(slope , gk)
        zero_cross = np.where(self.smooth_slope > 0)[0]
        if(len(zero_cross)>0):
            self.r_bg_index = zero_cross[0]
            self.r_bg = xdata[self.r_bg_index-1]
        else:
            self.r_bg_index = len(self.prof)-1
        # self.r_bg_index = np.max(self.beam/2, self.r_bg_index).astype('int') #r_bg at least 1 beam/2
            self.r_bg = xdata[self.r_bg_index-1]
        # print(slope)
    def plot_slope(self, ax,  field = None):
        if(field is not None): # convert x-axis to sky distance in parsec
            dist = field._get_sky_dist(self.dist)
            rbg = field._get_sky_dist([self.r_bg])[0]
        else: 
            dist = self.dist
            rbg = self.r_bg
        # print(self.slope)
        ax.plot(dist , self.slope, c = 'lightgray', lw=1, label = 'Slope')
        ax.plot(dist , self.smooth_slope ,c = 'k' , lw=1, label = 'Convolved slope')
        # ax.set_ylim(-0.3,0.3)
        ax.axhline(0,  lw=1, ls='--')
        # ax.set_xscale('log')
        ax.axvline(rbg , c = 'crimson' , lw=1)
        return ax
    
    

    def get_props(self, dist_modifier):
        if(self.fil_props is not None): return self.fil_props
        fil_intensity = self.prof[0]
        bg_intensity = self.prof[self.r_bg_index]
        fil_contrast = (fil_intensity - bg_intensity ) / bg_intensity
        dist = self.dist[np.where(self.dist<=self.r_bg)]
        pf = self.prof[np.where(self.dist<=self.r_bg)]
        lm  = trapezoid(x = dist_modifier(dist) , y = pf-bg_intensity)
        self.lmd = lm

        plm = plummer_fit_profile(self.dist, self.prof, self.err , self.r_bg, dist_modifier)

        fil_props = {
            'F_intensity' : fil_intensity , 
            'B_intensity' : bg_intensity , 
            'F_contrast' : fil_contrast,
            'R_bg' : self.r_bg, 
            'R_bg_index' : self.r_bg_index,
            'lmd' : lm ,
            'plm' : plm
        }
        self.fil_props = fil_props
        return fil_props
        
        

# @st.cache_data
def get_median(prof_list):
    # print('geting median')
    # message("getting median" , 2)
    max_length = max([len(p) for p in prof_list])
    new_prof_arr = []
    for p in prof_list:
        new_prof = np.ones(max_length)*-999
        new_prof[:len(p)] = p
        new_prof[new_prof==-999] = np.nan
        new_prof_arr.append(new_prof)
    new_prof_arr = np.asarray(new_prof_arr)
    med_prof = np.nanmedian(new_prof_arr , axis = 0)
    sigma = np.nanstd(new_prof_arr, axis =0)
    mad = 1*np.nanmedian(np.abs(new_prof_arr - np.nanmedian(new_prof_arr, axis=0)), axis=0)
    dist_max = np.arange(0, max_length , 1) 

    return np.asarray([dist_max , med_prof , mad , sigma])

# @st.cache_data
def get_median_tangent(cx, cy):
    coeff, _ , _, _, _ = np.polyfit(cx + 1, cy + 1, 1, full=True)
    m = -1.0 / coeff[0]
    return m


from sutra.profilerV2.plummer import plummer_fn
from scipy.optimize import curve_fit

from sutra.profilerV2.fitUtils import calc_red_chi , _curve_fit


class profGroup():
    '''
    Handles collection of filament profiles. For example : for a given filament, all the profiles within one beam-width can be grouped together
    '''
    def __init__(self, prof_arr , beam = 3, dist_modifier = None):
        self.prof_dicts = prof_arr
        self.line_profs = []
        for p in prof_arr:
            colden = p['col_dens']
            pixdist = p['pix_dist']
            prof = filProfile(pixdist, colden)
            prof.convolve_prof(beam)
            prof.get_center()
            prof.fold()
            self.line_profs.append(prof)

        
        self.beam = beam

        self.props = None
        self.dist_modifier = dist_modifier
        self.get_center()
        self.get_length()
        self.get_beam_slope()
        # self.med_prof_right = fitProf(*self.med_prof_right, beam)


    def get_center(self):
        '''
        center location of selected group of filaments in the image plane (x,y) pixel coordinates
        '''
        carr = np.asarray([p['cen'] for p in self.prof_dicts]).T
        self.gc = np.mean(carr, axis=1) #group center
        
        # self.low , self.high = 
        # return self.gc

    def get_beam_slope(self):
        cx, cy = np.asarray([p['cen'] for p in self.prof_dicts]).T
        self.m =  get_median_tangent(cx,cy)


    def get_length(self):
        '''
        using the start and end coordinates of the given filament group, the total length in pixels. If we have profiles at each pixel, then we could have just counted the number of pixels.
        '''
        dist = 0
        for p1,p2 in zip(self.prof_dicts[1:],self.prof_dicts[:-1] ):
            c1 , c2 = p1['cen'] , p2['cen']
            dist += np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        self.length = dist # length in pixel units
        return dist

    # def get_plummer_fit(self , dist):
        # if self.props is None : 


    def get_props(self, refresh=True):
        # if ~refresh :
        #     if self.props is not None : return self.props
        
        self.med_prof_right = get_median([p.prof_right for p in self.line_profs])
        self.med_prof_left = get_median([p.prof_left for p in self.line_profs])
        self.med_prof_all = get_median([p.prof for p in self.line_profs])
        self.med_prof_all[0] = self.med_prof_all[0] - np.abs(np.min([np.min(p.dist) for p in self.line_profs]))

        self.med_prof_right_fitter = fitProf(*self.med_prof_right, 
                                             beam=self.beam)
        self.med_prof_left_fitter = fitProf(*self.med_prof_left, 
                                            beam=self.beam)
        self.get_center()
        self.get_length()
        # if(self.props is not None or refresh) : return self.props
        # print('asasas')
        props_right = self.med_prof_right_fitter.get_props(self.dist_modifier)
        props_left = self.med_prof_left_fitter.get_props(self.dist_modifier)
        props = {
            'RF' : props_right ,  # dictionary of RIGHT side properties of median profile
            'LF' : props_left , #dictionary of LEFT side properties of median profile
            'Center' : self.gc  , 
            'Length' : self.length , 
            'LMD' : self.med_prof_right_fitter.lmd+self.med_prof_left_fitter.lmd
        }
        dy = np.sin(self.m)*props['RF']['R_bg'] 
        dx = np.cos(self.m)*props['RF']['R_bg']
        rh = props['Center']+[dx,dy]

        dy = np.sin(self.m)*props['LF']['R_bg'] 
        dx = np.cos(self.m)*props['LF']['R_bg']
        rl = props['Center']-[dx,dy]
        props['RH'] = rh #radius high
        props['RL'] = rl # radius low

        # p = 

        self.props = props
        return props
    

    def get_plummer_fit(self, refresh=True):
        # if ~refresh :
        #     if self.props is not None : return self.props


        if self.props is None:
            self.get_props()
        try:
            # message("BP1", 'd')
            props = self.props
            prof = self.med_prof_right
            max_r_indx = props['RF']['R_bg_index']
            r_r , r_profcd , r_prof_err = prof[0][:max_r_indx] , prof[1][:max_r_indx] , prof[3][:max_r_indx]
            r_param , r_pcov, r_modelcd , r_rc = _curve_fit(r_r , r_profcd , r_prof_err, beam=self.beam)

            prof = self.med_prof_left
            max_r_indx = props['LF']['R_bg_index']
            l_r , l_profcd , l_prof_err = prof[0][:max_r_indx] , prof[1][:max_r_indx] , prof[3][:max_r_indx]
            l_param , l_pcov, l_modelcd , l_rc = _curve_fit(l_r , l_profcd , l_prof_err, beam=self.beam)
            
            l , r = (l_profcd , l_prof_err , l_modelcd) , (r_profcd , r_prof_err , r_modelcd) 
            all_model = np.append( l[2][::-1] , r[2])
            all_prof = np.append( l[0][::-1] , r[0])
            all_err = np.append( l[1][::-1] , r[1])
            # print(all_model)
            # print(l_profcd, l_prof_err)
            n_params = 8
            if len(l_profcd) <=4 or len(r_profcd)<=4:
                n_params = 4
            all_red_chi = calc_red_chi(all_prof , all_model , all_err, n_params=n_params)

            self.plummer_params = (r_param , l_param)
            self.red_chi = (all_red_chi, r_rc, l_rc)
            self.fit_flag = 1
            self.model_left = l_modelcd
            self.model_right = r_modelcd
            # self.
        except Exception as e: 
            message(e, 'fe')

        fit_dict = {
                "Pi-R" :  r_param[2],
                "Pi-L" : l_param[2] ,
                "Rflat_R" : r_param[1] ,
                "Rflat_L" : l_param[1] ,
                "Nfil" : np.nanmean([l_param[0] , r_param[0]]),
                "Nfil_L" : l_param[0], 
                "Nfil_R" :r_param[0], 
                "Nbg_R" : r_param[3],
                "Nbg_L" :l_param[3] ,
                "red-chi" : all_red_chi,
                "red-chi-R" : r_rc , 
                "red-chi-L" :l_rc , 
            }
        return fit_dict    #, self.red_chi , self.plummer_params , self.plummer_err
    
    def plot_plummer(self):
        fig, ((ax1, ax3) , (ax2, ax4)) = plt.subplots(2, 2, 
                                       sharex='col', figsize=(6,5), height_ratios=[4, 2], sharey='row')
        prof = self.med_prof_left
        r = prof[0]
        rlabel = 'r (pixels)'
        ax1.errorbar(r, prof[1],yerr=prof[2],  ls=":", zorder = 1 )
        ax1.plot(r[:len(self.model_left)], 
                 self.model_left, c = 'k', zorder = 2, lw = 1,
                 )
        ax1.invert_xaxis()
        
        prof = self.med_prof_right
        r = prof[0]
        rlabel = 'r (pixels)'
        ax3.errorbar(r, prof[1],yerr=prof[2], label = 'Data', ls=":" , zorder = 1)
        ax3.plot(r[:len(self.model_right)], 
                 self.model_right, c = 'k', zorder = 2, lw=1 ,
                 label = f'model | $\chi^2 = {self.red_chi[0]:.2f}$')
        
        ax3.label_outer()
        # ax3.legent()
        # ax3.invert_xaxis()
        
        ax1.set_ylabel("$N(H_2)$ X ( $10^{21})$")

        ax2 = self.med_prof_left_fitter.plot_slope(ax = ax2)
        ax4 = self.med_prof_right_fitter.plot_slope(ax = ax4)
        ax4.label_outer()

        # ax2.legend(fontsize = 11 )
        ax2.set_xlabel(rlabel)
        ax2.set_ylabel("$dN(H_2) / dr$ X ( $10^{21})$")
        ax2.tick_params(axis='y', labelsize='small') #reduce fontsize

        ax3.legend(fontsize = 11 )
        plt.subplots_adjust(hspace=0.15, wspace = 0.)
        # plt.tight_layout()
