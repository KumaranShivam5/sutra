import numpy as np
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian1DKernel, convolve , convolve_fft , Gaussian2DKernel

from sutra.logger import message

def plummer_fn(x , peak , R , p , bg):
    '''
    Function to fit plummer profile
    params::
        x : distance array
        amplitude : amplitude
        R : R_flat 
        p : plummer index 
        bg : background
    TODO : include convolution with the beam : CONVOLVE MODEL profile
    '''
    nom =  peak - bg
    denom = (1 + (x/R)**2)**((p-1)/2) 
    prof_val = nom / denom + bg
    return prof_val

def get_dbgrad_local(dist, prof): # Filament radius where it meets background
        xdata, ydata = dist, prof
        dn = np.gradient(ydata)
        dr = np.gradient(xdata)
        # print(ydata)
        slope = dn/dr * (xdata/ydata) # slope

        dbgrad = (np.gradient(dn)/dr**2) * (xdata/ydata)


        gk = Gaussian1DKernel(stddev = 10/ 2.355) # gaussian kernel
        # self.slope = slope
        smooth_slope = convolve(slope , gk)
        dbgrad = convolve(dbgrad, gk)

        weird_mask = np.where(dbgrad>0, 1, 0)
        # print(weird_mask)

        return(weird_mask)
        # print(slope)





# TODO : update accoroding to the Arzoumanian
# updating fit
def plummer_fit_profile(pix_dist, prof, err, bgcut, dist_modifier=None):
    weird_mask = get_dbgrad_local(pix_dist, prof)
    
    ind = np.where(pix_dist<bgcut, 1, 0)
    indf = list(np.array(ind*weird_mask, dtype='int'))
    # indf = ind
    indf[0:5] = np.ones(5, dtype='int')
    truncdist = np.array([el for el,i in zip(dist_modifier(pix_dist), indf) if(i)])
    truncprof = np.array([el for el,i in zip(prof, indf) if(i)])
    truncerr = np.array([el for el,i in zip(err, indf) if(i)])
    maxamp = np.max(truncprof) - truncprof[-1]
    # print(maxamp, truncprof[-1])
    # fit_flag = 0

    try:
        popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False,  p0 = [maxamp, 0.03, 2.0, truncprof[-1]],
        bounds=([maxamp-0.8, 0.0001, 1, 0.001],[maxamp+0.8, 0.3, 4, truncprof[-1]+0.8]))
    except:
        try:
        # print('here')
            popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False,  p0 = [maxamp, 0.03, 2.5, truncprof[-1]],
                bounds=([maxamp-0.8, 0.0001, 1, 0.001],[maxamp+0.8, 0.3, 4, truncprof[-1]+0.8]))
        except:
            fit_flag=1
            popt = np.zeros(4)
            pcov = np.zeros((4,4))

    # print(popt)

    par_err = np.sqrt([pcov[i][i] for i in range(4)])

    if(popt[2]>3.999 or popt[2]<1):
        # print("limit warning(plummer index)")
        fit_flag = 1
    elif(popt[1]>0.2999):
        # print("limit warning(R flat)")
        fit_flag = 1
    else: fit_flag = 0

    ### refit more stringently?
    if(fit_flag==1):
        try:
            popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False, p0 = [maxamp, 0.03, 2, truncprof[-1]],
            bounds=([maxamp-0.01, 0, 1.95, truncprof[-1]-0.01],[maxamp+0.01, 0.3, 2.05, truncprof[-1]+0.01]))
            par_err = np.sqrt([pcov[i][i] for i in range(4)])
        except:
            fit_flag = 2


    return(popt, par_err, fit_flag)


# mod_parr, mod_par_errr, truncpixr = plummer_fit_profile(pg.med_prof_right.dist, pg.med_prof_right.prof,pg.med_prof_right.err, pg.med_prof_right.r_bg)
# mod_parl, mod_par_errl, truncpixl = plummer_fit_profile(pg.med_prof_left.dist, pg.med_prof_left.prof,pg.med_prof_left.err, pg.med_prof_left.r_bg)



# def fit_plummer(pixdist , colden , colden_err, bgcut , beam):
#     a_min , R_min , p_min , bg_min = np.min(profcd) , aquilaprof.hpbw.value / 4 , 0.5 , 0
#     a_max , R_max , p_max , bg_max = 2*np.max(profcd) , 2*aquilaprof.N_CUT_OFF_PIX  , 10 , np.max(profcd)
#     a_0 , R_0 , p_0, bg_0 = np.mean(profcd) , aquilaprof.N_CUT_OFF_PIX / 2 , 4 , np.mean(profcd)











## this function was used in the paper
def plummer_fit_profile_backup(pix_dist, prof, err, bgcut, dist_modifier=None):
    weird_mask = get_dbgrad_local(pix_dist, prof)
    
    ind = np.where(pix_dist<bgcut, 1, 0)
    indf = list(np.array(ind*weird_mask, dtype='int'))
    # indf = ind
    indf[0:5] = np.ones(5, dtype='int')
    truncdist = np.array([el for el,i in zip(dist_modifier(pix_dist), indf) if(i)])
    truncprof = np.array([el for el,i in zip(prof, indf) if(i)])
    truncerr = np.array([el for el,i in zip(err, indf) if(i)])
    maxamp = np.max(truncprof) - truncprof[-1]
    # print(maxamp, truncprof[-1])
    # fit_flag = 0

    try:
        popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False,  p0 = [maxamp, 0.03, 2.0, truncprof[-1]],
        bounds=([maxamp-0.8, 0.0001, 1, 0.001],[maxamp+0.8, 0.3, 4, truncprof[-1]+0.8]))
    except:
        try:
        # print('here')
            popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False,  p0 = [maxamp, 0.03, 2.5, truncprof[-1]],
                bounds=([maxamp-0.8, 0.0001, 1, 0.001],[maxamp+0.8, 0.3, 4, truncprof[-1]+0.8]))
        except:
            fit_flag=1
            popt = np.zeros(4)
            pcov = np.zeros((4,4))

    # print(popt)

    par_err = np.sqrt([pcov[i][i] for i in range(4)])

    if(popt[2]>3.999 or popt[2]<1):
        # print("limit warning(plummer index)")
        fit_flag = 1
    elif(popt[1]>0.2999):
        # print("limit warning(R flat)")
        fit_flag = 1
    else: fit_flag = 0

    ### refit more stringently?
    if(fit_flag==1):
        try:
            popt, pcov = curve_fit(plummer_fn, truncdist, truncprof, sigma = truncerr+0.00001, absolute_sigma=False, p0 = [maxamp, 0.03, 2, truncprof[-1]],
            bounds=([maxamp-0.01, 0, 1.95, truncprof[-1]-0.01],[maxamp+0.01, 0.3, 2.05, truncprof[-1]+0.01]))
            par_err = np.sqrt([pcov[i][i] for i in range(4)])
        except:
            fit_flag = 2


    return(popt, par_err, fit_flag)
