import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.colors as clr

from scipy.interpolate import interp1d


import matplotlib.colors as clr
from astropy import units as u
import copy

from astropy.convolution import Gaussian1DKernel, convolve
from tqdm import tqdm

cm = clr.LinearSegmentedColormap.from_list('binary',[(0, (0.1, 0.2, 0.5, 0.3)) , (1, 'red')])



# def _filament_props_worker(args):
#     """
#     args = (fil, fil_id, stride)

#     Returns a dictionary exactly like the one produced by the
#     original loop, or ``None`` if the filament raises an exception.
#     """
#     fil, fil_id, stride = args
#     try:
#         filprop = fil.get_fil_prop(stride=stride)          # heavy work
#         # ---- unpack centre coordinates ---------------------------------
#         x, y = filprop['C']['center'].T
#         filprop['C']['x']      = x
#         filprop['C']['y']      = y
#         filprop['C']['filID']  = [fil_id] * len(x)
#         return filprop
#     except Exception:                                     # keep behaviour of “except: continue”
#         return None





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


from scipy.integrate import simpson , trapezoid
from .plummer import plummer_fit_profile


class fitProf():
    '''
    This class handls one profile (ideally the median profile) and its properties
    '''
    def __init__(self, dist,prof,yerr, beam = 3):
        self.dist = dist
        self.prof = prof
        self.err = yerr
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
        gk = Gaussian1DKernel(stddev = self.beam/ 2.355) # gaussian kernel
        self.slope = slope
        self.smooth_slope = convolve(slope , gk)
        zero_cross = np.where(self.smooth_slope > 0)[0]
        if(len(zero_cross)>0):
            self.r_bg_index = zero_cross[0]
            self.r_bg = xdata[self.r_bg_index-1]
        else:
            self.r_bg_index = len(self.prof)-1
            self.r_bg = xdata[self.r_bg_index-1]
        # print(slope)
    def plot_slope(self, ax,  field = None):
        if(field is not None): # convert x-axis to sky distance in parsec
            dist = field.get_sky_dist(self.dist)
            rbg = field.get_sky_dist([self.r_bg])[0]
        else: 
            dist = self.dist
            rbg = self.r_bg
        # print(self.slope)
        ax.plot(dist , self.slope, c = 'lightgray', lw=1)
        ax.plot(dist , self.smooth_slope ,  lw=1)
        ax.set_ylim(-0.3,0.3)
        ax.axhline(0,  lw=1, ls='--')
        # ax.set_xscale('log')
        ax.axvline(rbg , c = 'crimson' , lw=1)
    
    

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
            'lmd' : lm ,
            'plm' : plm
        }
        self.fil_props = fil_props
        return fil_props
        
        

def get_median(prof_list):
    
    max_length = max([len(p) for p in prof_list])
    new_prof_arr = []
    for p in prof_list:
        new_prof = np.ones(max_length)*-999
        new_prof[:len(p)] = p
        new_prof[new_prof==-999] = np.nan
        new_prof_arr.append(new_prof)
    new_prof_arr = np.asarray(new_prof_arr)
    med_prof = np.nanmedian(new_prof_arr , axis = 0)
    mad = 1*np.nanmedian(np.abs(new_prof_arr - np.nanmedian(new_prof_arr, axis=0)), axis=0)
    dist_max = np.arange(0, max_length , 1) 

    return np.asarray([dist_max , med_prof , mad])


    

class profGroup():
    '''
    Handles collection of filament profiles. For example : for a given filament, all the profiles within one beam-width can be grouped together
    '''
    def __init__(self, prof_arr , beam = 3):
        self.profs = []
        self.prof_dicts = prof_arr
        for p in prof_arr:
            colden = p['col_dens']
            pixdist = p['pix_dist']
            prof = filProfile(pixdist, colden)
            prof.convolve_prof(beam)
            prof.get_center()
            prof.fold()
            self.profs.append(prof)

        self.med_prof_right = get_median([p.prof_right for p in self.profs])
        self.med_prof_left = get_median([p.prof_left for p in self.profs])
        self.med_prof_all = get_median([p.prof for p in self.profs])
        self.med_prof_all[0] = self.med_prof_all[0] - np.abs(np.min([np.min(p.dist) for p in self.profs]))
        self.med_prof_right = fitProf(*self.med_prof_right, beam=beam)
        self.med_prof_left = fitProf(*self.med_prof_left, beam=beam)

        self.props = None
        # self.med_prof_right = fitProf(*self.med_prof_right, beam)

    
    # def majority_cutoff(self):
    #     prof_arr = self.profs

    def _plot_one_prof(self,  dist , prof, proferr, ax=None , zorder = None):
        ax.plot(dist , prof,  c = 'crimson', linewidth = 3, marker = '.', zorder = zorder+1, alpha = 0.8)
        ax.errorbar(dist , y = prof, yerr = proferr, color='salmon' , alpha = 1 , zorder = zorder , elinewidth=2,lw = 0)
    
    def plot(self, ax=None, which = 'r', field = None):
        if (ax is None):
            ax = plt.subplot(111)
        if(which=='l'):
            # ax.set_title('Left Profile')
            for pr in self.profs: 
                if(field is not None): # convert x-axis to sky distance in parsec
                    dist = field.get_sky_dist(pr.dist_left)
                else: dist = pr.dist_left
                # field.get_sky_dist(self.dist)
                ax.plot(dist, pr.prof_left, c = 'navajowhite',lw=1, alpha = 0.6 , zorder  = 0)
            ax.plot([],[], lw=2, label = 'median profile', c='crimson')
            # ax.plot([] , [], c = '#fff399',lw=1, alpha = 1, label = 'Left Profile')
            if(field is not None): # convert x-axis to sky distance in parsec
                medprof_dist = field.get_sky_dist(self.med_prof_left.dist)
            else: medprof_dist = self.med_prof_left.dist
            self._plot_one_prof(dist = medprof_dist, prof = self.med_prof_left.prof, proferr = self.med_prof_left.err, ax=ax, zorder = len(self.profs)+1)
            # ax.legend(loc = 2,  fontsize = 11)
        elif(which=='r'):
            # ax.set_title('Right Profile')
            for pr in self.profs: 
                if(field is not None): # convert x-axis to sky distance in parsec
                    dist = field.get_sky_dist(pr.dist_right)
                else: dist = pr.dist_right
                ax.plot(dist, pr.prof_right,  c = 'navajowhite',lw=1, alpha = 0.6)
            # ax.plot([] , [], c = '#fff399',lw=1, alpha = 1, label = 'Right Profile')
            if(field is not None): # convert x-axis to sky distance in parsec
                medprof_dist = field.get_sky_dist(self.med_prof_right.dist)
            else: medprof_dist = self.med_prof_right.dist
            self._plot_one_prof(dist = medprof_dist, prof = self.med_prof_right.prof, proferr = self.med_prof_right.err, ax=ax, zorder = len(self.profs)+1)
        else:
            for pr in self.profs: 
                ax.plot(pr.dist, pr.prof, label='Profile', c = 'k', alpha = 0.1, lw=1)
            ax.plot(self.med_prof_all[0], self.med_prof_all[1], c = 'k', )
        # ax.set_xscale('log')
        # ax.set_xlabel('Distance from filament spine (pc)')
        ax.set_xscale('log')
        ax.set_ylabel(r'$N_{H_2}$ ($10^{21}cm^{-2}$)')
        ax.tick_params(axis="both", direction='in', length=2, which='both')

    def get_center(self):
        '''
        center location of selected group of filaments in the image plane (x,y) pixel coordinates
        '''
        carr = np.asarray([p['cen'] for p in self.prof_dicts]).T
        self.gc = np.median(carr, axis=1) #group center
        sloparr = np.asarray([p['slope'] for p in self.prof_dicts])
        self.m = np.mean(sloparr)
        # return self.gc
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


    def get_props(self, dist_modifier):
        self.get_center()
        self.get_length()
        if(self.props is not None) : return self.props
        # print('asasas')
        props_right = self.med_prof_right.get_props(dist_modifier)
        props_left = self.med_prof_left.get_props(dist_modifier)
        props = {
            'RF' : props_right ,  # dictionary of RIGHT side properties of median profile
            'LF' : props_left , #dictionary of LEFT side properties of median profile
            'Center' : self.gc  , 
            'Length' : self.length , 
            'LMD' : self.med_prof_right.lmd+self.med_prof_left.lmd
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


class Filament():
    def __init__(self, field , findx, stride=1):
        self.field = field
        self.findx = findx
        self.props = None
        self.stride = stride
    def get_sky_dist(self, pixdist):
        wcs = WCS(self.field.header)
        ps = u.pixel_scale((wcs.pixel_scale_matrix[1,1]*u.degree/u.pixel)) 
        tdist = (pixdist*u.pixel).to(u.rad, ps)*self.field.meta_info['distance']
        tdist = [t.value for t in tdist]
        return tdist
    
    def get_length(self):
        """
        Calculates the length of a filament and returns the number of profiles,
        the total pixel distance between profile centers, and the length in beam widths.

        This function calculates the length of a filament by summing the Euclidean distances
        between the centers of consecutive profiles within the filament.  It operates on the
        individual profiles *before* any profile grouping occurs.

        Returns:
            tuple: A tuple containing:
                - nprof (int): The number of profiles in the filament.
                - pix_dist (float): The total pixel distance between the centers of consecutive profiles.
                - beam_dist (float): The filament length expressed in units of the beam width (HPBW).
        
        Raises:
            TypeError: if `self.findx` is not a valid index for filaments in `self.field`.
            IndexError: if `self.findx` is out of range.
        """
        try:
            fil = self.field.get_filament(self.findx)
        except (TypeError, IndexError) as e:
            raise ValueError(f"Invalid filament index: {self.findx}.  Check that the index is valid for the current field.") from e
        
        fil = self.field.get_filament(self.findx)
        pix_dist = 0
        for p1,p2 in zip(fil[1:],fil[:-1] ):
            c1 , c2 = p1['cen'] , p2['cen']
            pix_dist += np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        nprof = len(fil)
        beam_dist = pix_dist / self.field.hpbw.value
        return nprof , pix_dist , beam_dist


    def get_fil_loc(self):
        """
        Determines the world coordinates of the start, center, and end points of a filament.

        This function retrieves the pixel coordinates of the start, middle, and end points
        of a filament and converts them to world coordinates (e.g., RA and Dec) using the
        field's WCS (World Coordinate System) information.

        Returns:
            tuple: A tuple containing the world coordinates (RA, Dec) of:
                - l1s (SkyCoord): Start point of the filament.
                - l2s (SkyCoord): Center point of the filament.
                - l3s (SkyCoord): End point of the filament.

        Raises:
            TypeError: if `self.findx` is not a valid index for filaments in `self.field`.
            IndexError: if `self.findx` is out of range.
            ValueError: If the filament does not have at least three profiles.
        """
        try:
            fil = self.field.get_filament(self.findx)
        except (TypeError, IndexError) as e:
            raise ValueError(f"Invalid filament index: {self.findx}. Check that the index is valid for the current field.") from e

        if len(fil) < 3:
            raise ValueError("Filament must have at least three profiles to determine start, center, and end points.")

        l1, l2, l3 = fil[0]['cen'], fil[int(len(fil) / 2)]['cen'], fil[-1]['cen']
        wcs = WCS(self.field.header)
        
        # Handle potential errors during coordinate transformation
        try:
            l1s = wcs.pixel_to_world(*l1, u.pixel)
            l2s = wcs.pixel_to_world(*l2, u.pixel)
            l3s = wcs.pixel_to_world(*l3, u.pixel)
        except Exception as e:  # Catch more general exceptions
            raise RuntimeError(f"Error transforming coordinates: {e}") from e

        return l1s, l2s, l3s


    def get_fil_prop(self,  stride = 1, NORM = 1e21): 
        if(self.props is not None) : return self.props
        if(self.stride):stride=self.stride
        fil = self.field.get_filament(self.findx)

        l1 , l2, l3 = fil[0]['cen'] ,fil[int(len(fil)/2)]['cen'] , fil[-1]['cen']
        wcs = WCS(self.field.header)
        l1s , l2s , l3s = wcs.pixel_to_world( *l1*u.pixel) , wcs.pixel_to_world( *l2*u.pixel ),wcs.pixel_to_world( *l3*u.pixel)
        #TODO : convert it into proper string
        self.loc = [l1s,l2s,l3s]

        beam = self.field.hpbw.value
        # pg_arr = [profGroup(f, beam=beam) for f in chunk_array(fil[int(2*beam):-int(2*beam)], int(beam*stride))]
        cent = []

        for p in fil:
            cent.append(p['cen'])
        cent = np.asarray(cent)
        # compute distances between centers. do cumulative sum. remainder it with the size of chunk required
        g = np.sqrt(np.square(cent[1:]-cent[:-1]).sum(axis=1)).cumsum() % (self.field.hpbw*stride).value
        mask = g[:-1]>g[1:]
        indx = np.arange(0, len(g)-1)
        chunk_locs = indx[mask]
        pg_arr = [profGroup(fil[c1:c2], beam = self.field.hpbw.value) for c1,c2 in zip(chunk_locs[:-1],chunk_locs[1:])]
        self.pg_arr = pg_arr
        

        props_arr = [p.get_props(dist_modifier = self.field.get_sky_dist) for p in pg_arr]
        # return props_arr
        rr_arr  = np.asarray([p['RF']['R_bg'] for p in props_arr])
        rl_arr  = np.asarray([p['LF']['R_bg'] for p in props_arr])
        FI  = [p['LF']['F_intensity'] for p in props_arr]
        RFC  = [p['RF']['F_contrast'] for p in props_arr]
        LFC  = [p['LF']['F_contrast'] for p in props_arr]
        RBI  = [p['RF']['B_intensity'] for p in props_arr]
        LBI  = [p['LF']['B_intensity'] for p in props_arr]

        Pr = [p['RF']['plm'][0][2] for p in props_arr] #plummer right
        Pr_e = [p['RF']['plm'][1][2] for p in props_arr] #plummer right error
        ffr = [p['RF']['plm'][2] for p in props_arr] #fit-flag-right

        Pl = [p['LF']['plm'][0][2] for p in props_arr] #plummer right
        Pl_e = [p['LF']['plm'][1][2] for p in props_arr] #plummer right error
        ffl = [p['LF']['plm'][2] for p in props_arr] #fit-flag-right
        ff = [ffr, ffl]


        RFLr = [p['RF']['plm'][0][1] for p in props_arr] #R-flat right
        RFLr_e = [p['RF']['plm'][1][1] for p in props_arr] #R-flat right error

        RFLl = [p['LF']['plm'][0][1] for p in props_arr] #R-flat right
        RFLl_e = [p['LF']['plm'][1][1] for p in props_arr] #R-flat right error
        
        Pltot_v = []
        Pltot_e = []
        RFLtot_v = []
        RFLtot_e = []
        # if both fits are good, just avg, if one of them is good take only that , if both are bad : sad
        for i in range(len(ffl)):
            if((ffl[i]+ffr[i] == 0) or (ffr[i]==1 and ffl[i]==1)): # avg 
                Pltot_v.append((Pr[i]+Pl[i])/2)
                Pltot_e.append((Pr_e[i]+Pl_e[i])/2)
                RFLtot_v.append((RFLr[i]+RFLl[i])/2)
                RFLtot_e.append((RFLr_e[i]+RFLl_e[i])/2)
            elif(ffl[i]==0): #ffl
                Pltot_v.append(Pl[i])
                Pltot_e.append(Pl_e[i])
                RFLtot_v.append(RFLl[i])
                RFLtot_e.append(RFLl_e[i])
            elif(ffr[i]==0): #ffr
                Pltot_v.append(Pr[i])
                Pltot_e.append(Pr_e[i])
                RFLtot_v.append(RFLr[i])
                RFLtot_e.append(RFLr_e[i])
            else: #nan
                Pltot_v.append(np.nan)
                Pltot_e.append(np.nan)
                RFLtot_v.append(np.nan)
                RFLtot_e.append(np.nan)

        Pltot_v  = np.asarray(Pltot_v)
        Pltot_e  = np.asarray(Pltot_e)
        RFLtot_e  = np.asarray(RFLtot_e)
        RFLtot_v  = np.asarray(RFLtot_v)


        mask = np.isnan(FI)
        rr_arr[mask] = np.nan
        rl_arr[mask] = np.nan
        cen = np.asarray([p['Center'] for p in props_arr])
        length = np.asarray([p['Length'] for p in props_arr])
        length = np.asarray(self.get_sky_dist(length))
        # return length
        # LMD = #linear mass density
        # ceny = [p.gc[1] for p in props_arr]
        LMD = np.asarray([p['LMD'] for p in props_arr]) # LMD in NH2/cm^2 *pc
        cm_to_pc = 3.2e-19 # cm to parsec
        mh = 2.8*1.6e-27 # hydrogen mass
        Ms = 1.989e30 # solar mass
        LMD = LMD*(NORM) * mh / (cm_to_pc**2 * Ms) # solar mass / parsec
        mass = np.asarray(length)*LMD
        wbg = np.asarray(self.get_sky_dist(rr_arr+rl_arr))
        fc = np.nanmean([RFC, LFC], axis=0)
        bi = np.nanmean([RBI,LBI], axis=0)
        # fi = np.nanmean(FI, axis=0)
        # loc = self.get_fil_loc()
        props = {
            'R' : {'FC' : np.asarray(RFC) , 'BI' : np.asarray(RBI) , 'Rbg' : self.get_sky_dist(rr_arr) , 'P' : Pr , 'P_e' : Pr_e , 'fit-flag' : ffr} , 
            'L' : {'FC' : np.asarray(LFC) , 'BI' : np.asarray(LBI) , 'Rbg' : self.get_sky_dist(rl_arr), 'P' : Pl , 'P_e' : Pl_e, 'fit-flag' : ffl},
            'C' : {'center' : cen ,'Length' : length , 'P-Index' : Pltot_v, 'RFL' : RFLtot_v, 'W_bg' : wbg , 'FI' : np.asarray(FI), 'LMD' : LMD , 'Mass' : mass, 'FC': fc , 'BI' : bi ,  'fit-flag-right' : ffr , 'fit-flag-left' : ffl } ,
            'Total' : {
                'Location' : self.loc ,#location of start, end and center ra-dec in deg
                'Length' : np.sum(length) , 
                'Area' : np.sum(np.nan_to_num(length, nan=0.0)*np.nan_to_num(wbg, nan=0.0)) , 
                'Mass' :    np.sum(np.nan_to_num(mass, nan=0.0)),
                'LMD_m' : np.nanmean(LMD) ,'LMD_e' : np.nanstd(LMD) ,
                'Wbg_m' : np.nanmean(wbg), 'Wbg_e' : np.nanstd(wbg), 
                'FC_m' :  np.nanmean(fc) , 'FC_e' : np.nanstd(fc), 
                'FI_m' :  np.nanmean(FI) , 'FI_e' : np.nanstd(FI), 
                'BI_m' : np.nanmean(bi) , 'BI_e' : np.nanstd(bi) , 
                'P_m' : np.nanmean(Pltot_v) ,  
                'P_e' : np.sqrt(np.nansum(Pltot_e**2)) / len(Pltot_e) , 
                'RFL_m' : np.nanmean(RFLtot_v) , 
                'RFL_e' : np.sqrt(np.nansum(RFLtot_e**2)) / len(RFLtot_e) , 

            }
        }
        # print('sdsd')
        self.props = props
        return props


def message(msg, type=1):
    tp = {
        1 : '[INFO] >> ' , 
        2 : '[PROCESS] >> ' , 
    }
    print(f'{tp[type]} {msg}')
    return None

def simplify_dict(list_of_dicts):
  """
  Converts a list of dictionaries with identical keys into a dictionary of lists.

  Args:
    list_of_dicts: A list of dictionaries where each dictionary has the same keys.

  Returns:
    A dictionary where keys are the keys from the input dictionaries and values 
    are lists containing the corresponding values from each dictionary in the input list.
    Returns an empty dictionary if the input list is empty.
    Returns None if the input list is invalid (e.g., dictionaries have inconsistent keys).
  """

  if not list_of_dicts:
    return {}

  # Check for consistent keys (optional but recommended for robustness)
  first_dict_keys = list_of_dicts[0].keys()
  for d in list_of_dicts:
    if d.keys() != first_dict_keys:
      print("Error: Dictionaries have inconsistent keys.")
      return None  # Or raise an exception if you prefer

  result = {}
  for key in first_dict_keys:
    result[key] = [d[key] for d in list_of_dicts]

  return result


from tqdm import tqdm
import pandas as pd


class propsMap():
    def __init__(self, field):
        self.field = field
        self.NFIL = np.max(field.filnum)
        # fil_array = []
        self.filaments = None
        self.fil_map_table = None
        self.fil_table = None


    def filament_table(self, stride=1):
        if self.fil_table is not None: return self.fil_table
        if(self.filaments is None):
            # print('collecting filaments')
            self.collect_filaments(stride)
        findices = np.arange(self.NFIL)
        message(f'Computing length of each filament')
        fil_lengths = np.asarray([fi.get_length() for fi in self.filaments])
        dfa = pd.DataFrame({
            'Findex' : findices , 
            'N_profiles' : fil_lengths[:, 0] , 
            'L_px' :  fil_lengths[:, 1] ,
            'L_beam' : fil_lengths[:, 2]
        }).set_index('Findex')
        ############ DEBUG ##################
        ###### > 0 condition DEBUG Only
        #####################################
        dfa['L>2B'] = dfa['L_beam']>=1
        # return dfa
        fti = dfa[dfa['L>2B']].index.to_list() #'L>3B true' index
        # fti = 
        fp = []
        message(f'Computing properties of filaments having length > 3xbeam : {len(fti)}')
        for f in tqdm(np.asarray(self.filaments)[fti]):
            fp.append(f.get_fil_prop()['Total'])
        df = pd.DataFrame(simplify_dict(fp))
        df['Findex'] = fti
        df = df.set_index('Findex')
        df = pd.merge(dfa, df , left_index=True, right_index=True, how = 'left')
        self.fil_table  = df
        return df

    def collect_filaments(self, stride = 1):
        message(f'Total number of filaments : {self.NFIL}', 2)
        message(f'Collecting all filaments')
        fil_array = []
        for i in tqdm(range(self.NFIL)):
            fil_array.append(Filament(self.field, i, stride=stride))
        self.filaments = fil_array

    def get_props_maps(self, stride = 1, refresh = False):
        if self.filaments is None:
            self.collect_filaments(stride = 1)
        if self.fil_table is None: self.filament_table()
        mask = self.fil_table[self.fil_table['L>2B']].index.to_list()
        if self.fil_map_table is None or refresh:
            filprops = []
            i = 0
            for fil in tqdm(np.asarray(self.filaments)[mask]):
                # orion = Filament(pol, findx, stride = 1)
                filprop = fil.get_fil_prop(stride = 1)
                try:
                    x ,y = filprop['C']['center'].T
                    filprop['C']['x'] = x
                    filprop['C']['y'] = y
                    filprop['C']['filID'] = [mask[i]]*len(x)
                    # del filprop['C']['center'] 
                    filprops.append(filprop)
                except : continue
                i+=1
            dfl = []
            for f in filprops:
                ftmp = copy.deepcopy(f['C'])
                del ftmp['center']
                dfl.append(pd.DataFrame(ftmp))
                del ftmp
            df = pd.concat(dfl)
            df = df[df['Length'] > 0]
            self.fil_map_table  = df
            return self.fil_map_table
        else: return self.fil_map_table 


    def global_props(self):
        print('sdsd')
        LRbg , RRbg, RFC, RBI, FI, LBI, LFC, cen , lmd = [], [],[],[],[],[],[], [], []
        for fi in self.filaments:RRbg.extend(fi.props['R']['Rbg'])
        for fi in self.filaments:LRbg.extend(fi.props['L']['Rbg'])
        for fi in self.filaments:RFC.extend(fi.props['R']['FC'])
        for fi in self.filaments:LFC.extend(fi.props['L']['FC'])
        for fi in self.filaments:FI.extend(fi.props['C']['FI'])
        # for fi in orionA.filaments:LFI.extend(fi.props['L']['FI'])
        for fi in self.filaments:RBI.extend(fi.props['R']['BI'])
        for fi in self.filaments:LBI.extend(fi.props['L']['BI'])
        for fi in self.filaments:cen.extend(fi.props['C']['center'])
        for fi in self.filaments:lmd.extend(fi.props['C']['LMD'])

        props = {
            'R' : {'FC' : RFC , 'BI' : RBI , 'Rbg' : RRbg} , 
            'L' : {'FC' : LFC , 'BI' : LBI , 'Rbg' : LRbg},
            'C' : {'center' : cen , 'W_bg' : np.sum(RRbg,LRbg) , 'FI' : FI, 'LMD' :lmd}
        }
        return props