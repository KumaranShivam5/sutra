

from sutra.file_io import load_fits
from sutra.tracer.modifiers import remove_nan

from astropy.io import fits
from sutra.measurement import local_field




# from sutra.model import load_predictor

from sutra.tracer.predictor import filamentIdentifier as FID


class cloud:
    def __init__(self, input_file, meta_info = None) -> None:
        cd_array = load_fits(input_file)
        cd_array.data = remove_nan(cd_array.data) 
        self.cd = cd_array
        self.cd_header = cd_array.header   
        if meta_info is None:
            meta_info = {
                'distance' : 1800, # 140 pc for Taurus , 260 for musca
                'beam' : 36.4 , # arcsec 
                'radial-cutoff' : 0.3 , 
            } 
        self.meta_info = meta_info
        
    
    def find_filament(self, model='HiGAL', save_file = None, window_overlap_frac = 0.95):
        prob_map  = FID(predictor_name = model).predict(self.cd.data, window_overlap_frac = window_overlap_frac, n_jobs = 1)
        self.prob_map = prob_map
        probff = fits.HDUList([fits.PrimaryHDU(data = prob_map, header=self.cd_header)])
        self.local_field = local_field(self.cd, prob_map , meta_info = self.meta_info)
        if save_file:
            probff.writeto(f'{save_file}', overwrite=True)
        return probff
    
    def skeleton_map(self, threshold = 0.1, save_file = None):
        self.local_field.apply_skl_threshold(threshold)
        skeleton = self.local_field.radprof.skel
        skeletonff = fits.HDUList([fits.PrimaryHDU(data = skeleton*1, header=self.cd_header)])
        if save_file:
            skeletonff.writeto(f'{save_file}', overwrite=True)
        return skeletonff
    
    def all_filament_props(self, profile_pixel_stride = 3, beam_group_stride = 0.5):
        print("[INFO] >>> Extracting Radial profiles")
        self.local_field.process_skeleton(ks = 6, stride = profile_pixel_stride , reorder = 15)
        self.local_field.extract_rad_prof()
        fil_table =  self.local_field.get_filament_table()
        # musca_fil_table = musca_fil_table.drop(columns = ['Location']) 
        self.filament_table = fil_table
        props_df = self.local_field.get_filament_prop_map(stride = beam_group_stride , refresh=True)
        props_df['rad_pix']  = self.local_field.radprof.pc_to_pixel(props_df['W_bg']/4)
        self.props_df = props_df

        return {"Local Filament properties" : fil_table , "Global Filament Properties" : props_df}
