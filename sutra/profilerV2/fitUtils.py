from scipy.optimize import curve_fit
from sutra.profilerV2.plummer import plummer_fn 
import numpy as np
from sutra.logger import message

def calc_red_chi(data , model , sigma , n_params = 4):
    try:

        s = np.nansum(((model-data) / sigma)**2)
        
        # TODO : what if len(data) < n_params 
        red_chi  = s / (len(data)-n_params)
    except Exception as e:
        message(e , 'e')
        return np.nan 

    return red_chi

def _curve_fit(r , profcd , prof_err, beam):
    def _set_nan():
        param = 4*[np.nan]
        pcov = 4*[np.nan]
        rc = np.nan
        modelcd = [np.nan]*len(profcd)
        return param , pcov ,  modelcd,  rc
    try:
        if len(profcd) <= 4:
            # raise ValueError("Data points less than Number of parameters")
            message('Data points less than number of parameters ', 'e')
            return _set_nan()
    # print('inside curve fit')
        # message("breakpoint 7" , 'd')
        # print(profcd)
        # a_min , a_max, a_0 = np.min(profcd) , profcd[0] + prof_err[0] , profcd[0]
        a_min , a_max, a_0 = profcd[0]-profcd[0]*0.001, profcd[0]+profcd[0]*0.001, profcd[0]
        R_min , R_max , R_0 = beam / 2 ,  2*len(profcd) , beam
        p_min , p_max , p_0 = 1 , 6 , 4
        bg_min , bg_max , bg_0 = 0 ,  np.mean(profcd) , np.min(profcd) 
        # message("breakpoint 8" , 'd')

        # message("breakpoint 5" , 'd')

        param , pcov = curve_fit(plummer_fn, r , profcd, 
                        p0=[a_0 , R_0 , p_0, bg_0 ], 
                        bounds=([a_min , R_min , p_min , bg_min],[a_max , R_max , p_max , bg_max ]), 
                        sigma=prof_err, method='trf')
        modelcd = plummer_fn(r , *param)
        rc = calc_red_chi(profcd , modelcd , prof_err)
        # message("breakpoint 4" , 'd')

        if rc>50: # cases when model at 0 goes way beyond the obs

            R_max = max(r)
            R_0 = max(r)-1
            param , pcov = curve_fit(plummer_fn, r , profcd, 
                        p0=[a_0 , R_0 , p_0, bg_0 ], 
                        bounds=([a_min , R_min , p_min , bg_min],[a_max , R_max , p_max , bg_max ]), 
                        sigma=prof_err, method='trf')
            modelcd = plummer_fn(r , *param)
            rc = calc_red_chi(profcd , modelcd , prof_err)
    except Exception as e:
        message(e, 'e')
        return _set_nan()
    # message("breakpoint 1" , 'd')
    # print(param)
    return param , pcov , modelcd , rc
