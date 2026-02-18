# CLI for quickly generating filament map and filament skeleton map


from sutra.file_io import load_fits

from sutra.plots import make_wcs_good
from sutra.tracer.modifiers import remove_nan



from sutra.tracer.predictor import filamentIdentifier as FID

from astropy.io import fits
from sutra.measurement import local_field

import numpy as np
import argparse
from sutra.model import predictors_dict

def main():

    meta_info = {
                'distance' : 1800, # 140 pc for Taurus , 260 for musca
                'beam' : 36.4 , # arcsec 
                # 'radial-cutoff' : meta_info_block.number_input("Radial profile cutoff (pc)", value = 0.2)# 0.2 pc , 
                'radial-cutoff' : 0.3 , 
                # 'prob-thresh' :
            }




    # import argcomplete

    model_names = list(predictors_dict.keys())
    # print(model_names)
    # sdsdsd
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cd_file", dest="input_file", required=True, help="Path to the input Column density map file.")
    parser.add_argument("-s", "--skl_output", dest="skl_output_file", required=True, help="Path to the output Skeleton map.")
    parser.add_argument("-p", "--model_output", dest="model_output_file", required=True, help="Path to the output Skeleton map.")

    threshold = None 
    parser.add_argument("-t", "--threshold", dest="threshold",required = False, help="Threshold to convert model output to skeleton map")

    parser.add_argument("-m", "--model", dest="model", required=False, help="Threshold to convert model output to skeleton map")


    # # parser.add_argument("-p", "--plot", dest="plot", help="Plot image ?", required=False, action="store_true",)


    # # Parse the arguments from the command line
    args = parser.parse_args()

    # # Access the arguments
    input_file = args.input_file
    skl_output_file = args.skl_output_file
    model_output_file = args.model_output_file

    uploaded_file = f'{input_file}'
    cd_array = load_fits(uploaded_file)
    cd_array.data = remove_nan(cd_array.data)  

    cd_header = cd_array.header    
    prob_map  = FID(predictor_name='HiGAL').predict(cd_array.data , 'HiGAL')

    local_field = local_field(cd_array, prob_map , meta_info = meta_info)
    if threshold is None:
        threshold = float(input("Enter skeletonisaiton threshold (0,1) : "))
    # prob_thresh = 0.1
    local_field.apply_skl_threshold(threshold)
    skeleton = local_field.radprof.skel
    skeletonff = fits.HDUList([fits.PrimaryHDU(data = skeleton*1, header=cd_header)])
    skeletonff.writeto(f'{skl_output_file}', overwrite=True)


    probff = fits.HDUList([fits.PrimaryHDU(data = prob_map, header=cd_header)])
    probff.writeto(f'{model_output_file}', overwrite=True)





if __name__ == "__main__":
    main()