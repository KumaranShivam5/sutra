import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pandas as pd 
# import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gd
import math

# from fil_finder import FilFinder2D
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D 
from astropy.coordinates import SkyCoord
from astropy import units as u

        # from astropy.wcs import WCS
import os

def load_object(filename):
    import dill as pickle
    with open(filename, 'rb') as outp:
        return pickle.load(outp)


def save_object(obj , filename):
    import dill as pickle
    try:
        with open(filename, 'wb') as outp:
            pickle.dump(obj , outp)
        print(f"{obj} saved to file {filename}")
    except pickle.PickleError as e:
        print(f"Error occured in saving the prediction model : {e}")

class astroImage():
    def __init__(self, HDU, hdu_loc = 0):
        if(type(HDU)==type('str')):
            print('inside str')
            hdu = fits.open(HDU)[hdu_loc]
            self.hdu = hdu
            self.wcs = WCS(hdu.header)
        else:
            self.hdu = HDU
            self.wcs = WCS(HDU.header) 

    def plot_image(self, scale='log'):
        ax = plt.subplot(111, projection = self.wcs)
        im1 = plt.imshow(self.hdu.data, cmap='inferno', norm=scale)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel(' ')
        lat.set_axislabel(' ')
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        lon.set_major_formatter("d.dd")
        lat.set_major_formatter("d.dd")
        lon.set_ticklabel(exclude_overlapping=True)
        lon.set_axislabel("DEC(J2000)")
        lat.set_axislabel("DEC(J2000)")
        # ax6.coords[0].set_axislabel("RA(J2000)", rotation=45)
        cbar = plt.colorbar(im1)
        cbar.set_label(r"$Log_{10}(N(H_2))$")
        plt.show()

    def convert_distance(self, dist):
# def save_object(obj, filename):
#     import pickle
#     print('SAVING the object')
#     with open(filename, 'wb') as outp:
#         pickle.dump(obj, outp,)



        if(dist.unit=='arcsec'):
            scale = self.wcs.proj_plane_pixel_scales()[0].to(dist.unit)/(1*u.pixel)
            return dist/scale
        elif(dist.unit=='pix'):
            scale = self.wcs.proj_plane_pixel_scales()[0]/(1*u.pixel)
            return dist*scale
    
    def get_crop(self, loc, size, plot=False):
        loc = SkyCoord(*loc)
        size = size*u.pixel
        cut = Cutout2D(self.hdu.data, position = loc, size = size, wcs = self.wcs )
        hdul = fits.HDUList([fits.PrimaryHDU(data = cut.data, header = cut.wcs.to_header())])
        if(plot):
           tmp_hdu = astroImage(hdul[0]).plot_image(scale='log')
        return hdul



        # mon_cd = mon_cut.data

# FNAME = 'tmp-data/HGBS_aquilaM2_hires_column_density_map.fits'
# aquila = astroImage(FNAME)
# aquila.convert_distance(1*u.arcsec)
# # aquila.convert_distance(1*u.pixel)
# cut = aquila.get_crop(loc = (277.4861671*u.deg , -1.8320213*u.deg), size = 1500 , plot=True)
# # cut.plot_image()
# cut.writeto('tmp-data/aquila-cropped.fits')


# def save_object(obj, filename):
#     import pickle
#     print('SAVING the object')
#     with open(filename, 'wb') as outp:
#         pickle.dump(obj, outp,)




class obsFieldHandler:
    '''
    Class for handling the data processing, subdivision saving and loading of observation field of arbitrary size into required format.

    ...

    Attributes
    ----------
    name : str
        name of the field, example 'Polaris'
    cd: FITS HDUList
        FITS HDU list where the primary HDU contains the column density data, or the main observation data
    fil: None, Any
        FITS file containing the filament skeleton as PrimaryHDU.
        IF the application of this object is not to train the model, rather to apply, skip this.
    global_normalizer : array
        Array of functions to be applied as global preprocessor
    local_normalizer : array
        Array of functions to be applied as local preprocessor on individual chunks
    skeleton_modifier : array
        Array of functions to be applied on the filament skeleton

    '''
    def __init__(self, name, cd, fil=None, global_normalizer = [], local_normalizer=[] , skeleton_modifier = []) -> None:
        '''
        cd : FITS file containing the cloud column density as PrimaryHDU
        fil : FITS file containing the filament skeleton as PrimaryHDU

        Note
        -----
        Only the local_normalizer is aplied to the individual segments.
        '''
        self.og_cd = cd 
        if(fil is not None):
            self.og_fil = fil 
        self.field_header = cd[0].header
        cd = np.asarray(cd[0].data)
        # fil = np.asarray(fil[0].data)
        if(global_normalizer):
            for g in global_normalizer:
                cd = g(cd)
       
        self.cd = cd
        self.global_normalizer = global_normalizer 
        self.local_normalizer = local_normalizer 
        self.skeleton_modifier = skeleton_modifier 

        if(fil is not None):
            fil = np.asarray(fil[0].data)
            if(skeleton_modifier):
                for s in skeleton_modifier:
                    fil = s(fil)
            self.fil = fil 
        self.fil = fil
        self.name = name
        

        # cd = np.asarray(cd[0].data)
        # fil = np.asarray(fil[0].data)
        # # fil = np.nan_to_num(fil, nan=0,)
        # # cd = np.nan_to_num(cd, nan=0,)
        # self.name = name
        
        # if(global_normalizer):
        #     for g in global_normalizer:
        #         cd = g(cd)
        # if(skeleton_modifier):
        #     for s in skeleton_modifier:
        #         fil = s(fil)
        # self.fil = fil 
        # self.cd = cd
        # self.global_normalizer = global_normalizer 
        # self.local_normalizer = local_normalizer 
        # self.skeleton_modifier = skeleton_modifier 




    def get_cd(self):
        return self.cd
    def get_fil(self):
        return self.fil

    def clear_data(self):
        self.cd = None
        self.fil = None

    def create_im_chunks(self , CHUNK_SIZE = 256 ,  DEL_PIX=110, chunk_loc = 'train_data', fil_only=True, wcs = False):
       
        SIZE_X , SIZE_Y = self.cd.shape
        x_chunks = int((SIZE_X - CHUNK_SIZE) / DEL_PIX)
        y_chunks = int((SIZE_Y -CHUNK_SIZE) / DEL_PIX)
        n_chunks = x_chunks * y_chunks

        print(f'[INFO] Provided image size : {self.cd.shape}')
        print(f'[INFO] generating segments of size : {CHUNK_SIZE}x{CHUNK_SIZE}')
        print(f'[INFO] Total number of images to be generated ({x_chunks} x {y_chunks}): {n_chunks}')
        N_img = 0
        if(fil_only):
            print('[TASK] Generating only the segments containing a filament..')
        
        if(chunk_loc=='temp'):
            os.system('rm -r temp')
            os.system('mkdir temp')
            os.system('mkdir temp/CD')
            os.system('mkdir temp/skeleton')
            # os.system('mkdir temp/skeleton')
        for x in tqdm(range(x_chunks)[:]):
            for y in range(y_chunks)[:]:
                x_start, x_end = int(x * DEL_PIX ) ,  int(x * DEL_PIX + CHUNK_SIZE)
                y_start, y_end = int(y * DEL_PIX)  ,  int(y * DEL_PIX + CHUNK_SIZE)
            
                if(fil_only):
                    if(len(np.unique(self.fil[x_start:x_end, y_start:y_end]))>1):
                        # print('sdsdsddsd' )
                        # print(len(np.unique(self.fil[x_start:x_end, y_start:y_end])))
                        # N_img +=1
                        # self.regional_norm_segment(x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE ,  DEL_PIX)
                        N_img += self._save_segment(x,y, chunk_loc, CHUNK_SIZE ,  DEL_PIX, wcs = wcs)
                else:
                    print('in else')
                    N_img += self._save_segment(x,y, chunk_loc, CHUNK_SIZE,  DEL_PIX, wcs = wcs)
                # else: print(np.isnan(self.cd[x_start:x_end, y_start:y_end]).sum())
    
        print(f'[INFO] Total Image generated: {N_img}')

    def _save_segment(self,x,y, chunk_loc, CHUNK_SIZE,  DEL_PIX , wcs = False):
        '''
        this time saving segments in binary format for more efficiency
        '''
        seg_cd = self._get_segment(self.og_cd[0], x,y, CHUNK_SIZE, DEL_PIX, preserve_wcs = wcs)
        # return seg_cd
        if(wcs == False):
            # print(self.cd)
            if(int(np.isnan(seg_cd).sum())>0):
                return 0
            else:
                if(self.local_normalizer):
                    for l in self.local_normalizer:
                        seg_cd = l(seg_cd)
                np.save(f'{chunk_loc}/CD/{self.name}_{x}_{y}.npy', seg_cd ,)
                del seg_cd

                # print(self.fil is not None)
                if(self.fil is not None):
                    seg_fil = self._get_segment(self.og_fil[0], x,y, CHUNK_SIZE, DEL_PIX)
                    np.save(f'{chunk_loc}/skeleton/fil_{self.name}_{x}_{y}.npy', seg_fil ,)
                    del seg_fil
                return 1
        else:
            seg_cd_data = seg_cd[0].data
            if(int(np.isnan(seg_cd_data).sum())>0):
                return 0
            else:
                if(self.local_normalizer):
                    for l in self.local_normalizer:
                        seg_cd_data = l(seg_cd_data)
                # np.save(f'{chunk_loc}/CD/{self.name}_{x}_{y}.npy', seg_cd ,)
                seg_cd[0].data = seg_cd_data
                seg_cd.writeto(f'{chunk_loc}/CD/{self.name}_{x}_{y}.fits', overwrite = True)
                del seg_cd
                # return 1

                # print(self.fil is not None)
                if(self.fil is not None):
                    seg_fil = self._get_segment(self.og_fil[0], x,y, CHUNK_SIZE, DEL_PIX, preserve_wcs = wcs)
                    # seg_fil_data = seg_fil[0].data
                    seg_fil.writeto(f'{chunk_loc}/skeleton/fil_{self.name}_{x}_{y}.fits',  overwrite = True)
                    del seg_fil
                return 1

        


    def _get_segment(self, field,x,y, CHUNK_SIZE, DEL_PIX, preserve_wcs = False):
        '''
        return a segment of given field
        '''
        try:
            x_start, x_end = int(x * DEL_PIX ) ,  int(x * DEL_PIX + CHUNK_SIZE)
            y_start, y_end = int(y * DEL_PIX)  ,  int(y * DEL_PIX + CHUNK_SIZE)


            if(preserve_wcs):

                # image = self.og_cd[0].copy()
                image = field.copy()
                # return image
                # image = fits.PrimaryHDU(data = image.data, header = self.field_header)
                ## Inverting x and y position as in fits file axis 1 is vertical and
                # axis 2 is horizontal
                position = ((x_start+x_end)/2 , (y_start+y_end)/2)
                # position = 
                # x_start ,y_start = position 
                position = np.flip(position)
                cutout = Cutout2D(image.data , position = position , size = CHUNK_SIZE*u.pixel, wcs=WCS(image.header))
                image.data = cutout.data
                # added this to correctly add chunk wcs
                image.header.update(cutout.wcs.to_header()) 
                hdul = fits.HDUList([image])
                # print(hdul)
                return hdul
            # print()
            cd_arr = field.data[x_start:x_end, y_start:y_end]
            # fil_arr = self.fil[x_start:x_end, y_start:y_end]
            if(cd_arr.shape==(CHUNK_SIZE,CHUNK_SIZE)):
                # print('returning true value')
                return cd_arr
            else:
                # print('returning zero value')
                return np.zeros((CHUNK_SIZE, CHUNK_SIZE)) , np.zeros((CHUNK_SIZE, CHUNK_SIZE))

            
        except Exception as e:
            print('>>>> [ERROR]', x,y, '---', e)
            return np.zeros((CHUNK_SIZE, CHUNK_SIZE)) , np.zeros((CHUNK_SIZE, CHUNK_SIZE))

    def _create_wcs_segment(self, field, x,y, CHUNK_SIZE, DEL_PIX, fname=None):
 
        '''
        creates segments and preserve the WCS coords
        '''
        x_start = int(x * DEL_PIX )
        y_start = int(y * DEL_PIX)
        image = field.copy()
        wcs = WCS(image.header)
        ## Inverting x and y position as in fits file axis 1 is vertical and
        # axis 2 is horizontal
        x_start ,y_start = position 
        position = x_start ,y_start
        cutout = Cutout2D(image.data , position , CHUNK_SIZE, wcs=wcs)
        image.data = cutout.data
        hdul = fits.HDUList([image])
        return hdul

        if(sum(sum(~np.isnan(image.data)))):
            image.header.update(cutout.wcs.to_header())
            hdul = fits.HDUList([image])
            if(fname):
                hdul.writeto(fname,overwrite=True)
                hdul.close()
            else:
                return hdul
            return 1 
        else:
            return 0






from tensorflow import keras
class trainDataLoader(keras.utils.Sequence):


    def __init__(self, batch_size , input_img_paths, target_img_paths,):
        
        self.input_img_paths = input_img_paths 
        self.target_img_paths = target_img_paths
        if(batch_size==-1):
            batch_size = len(input_img_paths)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.input_img_paths) / self.batch_size)

    def __getitem__(self, idx):
        if(idx > len(self)):
            raise ValueError('Index does not exist')
        img_paths = self.input_img_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        target_paths = self.target_img_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        x_temp, y_temp = [] ,  [] 
        for im , tg in zip(img_paths, target_paths):
            x_temp.append(np.load(im))
            y_temp.append(np.load(tg))
        x_temp = np.asarray(x_temp)
        y_temp = np.asarray(y_temp)
        return x_temp,y_temp


def main():
    return None

if __name__ == "__main__":
    argcomplete.autocomplete(parser)
    # main()