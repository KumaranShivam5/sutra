import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pandas as pd 
# import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gd
import math

from fil_finder import FilFinder2D
from astropy.wcs import WCS 



def flatten_image(hdu_file , flatten_percent=95):
    # print(isinstance(hdu_file, str), type(hdu_file))
    if(isinstance(hdu_file, str)):
        # print('inside if')
        hdu = fits.open(hdu_file)
    else:
        hdu = hdu_file
    print('hdu type:', type(hdu))
    fil = FilFinder2D(hdu[0])
    fil.preprocess_image(flatten_percent=flatten_percent)
    plt.subplot(121)
    plt.imshow(fil.image.value, origin='lower')
    plt.title("Image")
    plt.subplot(122)
    plt.imshow(fil.flat_img.value, origin='lower')
    plt.title("Flattened Image")
    plt.tight_layout()
    plt.show()
    mod_fit = fits.PrimaryHDU(data = fil.flat_img.value, header = fil.header)
    return mod_fit




def plot_field(hdu_file, **kwargs):
    if(type(hdu_file)=='str'):
        hdu = fits.open(hdu_file)
    else:
        hdu = hdu_file
    if('hdu_index' in kwargs):
        hdu_index = kwargs['hdu_index']
        hdu = hdu[hdu_index]
    else:
        hdu_index = 0
        hdu = hdu
    wcs = WCS(hdu.header)
    plt.subplot(projection = wcs)
    if('cmap' in kwargs):
        cmap = kwargs['cmap']
    else:
        cmap = 'inferno'
    plt.imshow(hdu.data, cmap=cmap)
    plt.show()





def normalize_CD(cd):
    '''
    for now doing simmple normalisation.
    need to implement histogram equalisation
    Hopefully global normalisation will reduce the inter-image brightness difference
    # overall normalisation does not work at all, somehow does not give any output prediction (figured out, somehow with nan in the array, np.amax does not work)
    '''
    print('[TASK]>> Doing Global Normalisation. max of cd', np.amax(cd))
    cd = cd / np.amax(cd)
    return cd

def histogram_eq(cd):
    print('[TASK]>>> Doing Histogram Equilazation')
    img_eq = np.sort(cd.ravel()).searchsorted(cd)
    return img_eq

from fil_finder import FilFinder2D
from astropy.io import fits


class Cloud:
    def __init__(self , name , cd , fil, global_norm = 0, norm = 'local', flatten=0):
        '''
        cd : FITS file containing the cloud column density as PrimaryHDU
        fil : FITS file containing the filament skeleton as PrimaryHDU
        '''
        cd = np.asarray(cd[0].data)
        fil = np.asarray(fil[0].data)
        fil = np.nan_to_num(fil, nan=0,)
        cd = np.nan_to_num(cd, nan=0,)
        self.name = name
        #
        if(global_norm):
            self.cd = global_norm(cd)
        else: self.cd = cd
        if (flatten>0):
            self.cd = self.flatten_cloud(flatten)

        self.fil = fil
    def flatten_cloud(self,flatten):
        fil = FilFinder2D(self.cd)
        fil.preprocess_image(flatten_percent=flatten)
        return fil.flat_img.value

    def get_cd(self):
        return self.cd
    def get_fil(self):
        return self.fil

    def plot_cloud(self, plot_level=1):
        '''
        plots cloud column density and filament skeleton
        '''
        # fig , ax = plt.subplots(nrows=2, ncols=2 , figsize=(12,10))
        # ax = np.ravel(ax)

        fig = plt.figure(figsize=(12,10))
        spec = fig.add_gridspec(nrows=2, ncols=2 , height_ratios = [3,1],)
        print('np.min', np.min(self.cd))
        ax0 = fig.add_subplot(spec[0,0])
        im = ax0.imshow(np.ma.log10(self.cd), origin='lower' , cmap='inferno_r')
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right" , size='5%', pad=0)
        plt.colorbar(im , cax=cax)

        ax1 = fig.add_subplot(spec[0,1])
        im = ax1.imshow(((self.fil)), origin='lower' , cmap='gray')

        ax2 = fig.add_subplot(spec[1,0])
        ax2.hist(np.ndarray.flatten((np.ma.log10(self.cd))))

        v , n = np.unique(self.fil[self.fil!=0] , return_counts=True)
        ax3 = fig.add_subplot(spec[1,1])
        ax3.bar(v,n)
        plt.tight_layout()
        plt.show()

    def plot_cd_hist(self):
        '''
        Plot the histogram of the Column density
        '''
        fig , ax = plt.subplots(nrows=1, ncols=2 , figsize=(10,6))
        ax = np.ravel(ax)
        ax[0].hist(np.log(self.cd))
        plt.show()

    def create_im_chunks(self , CHUNK_SIZE = 256 ,  DEL_PIX=110, chunk_loc = 'train_data', fil_only=True):
        # added the argument fil_only, so that if the given cloud is new and we do not have the filaments,rather we need to apply our model to get the filaments, we will simply generate all possible chunks without careing if the chunk have filament or not. for the trainig data, we create those chunks only, those have a filament
        SIZE_X , SIZE_Y = self.cd.shape
        # print(self.cd.shape)
        # print(type(CHUNK_SIZE))
        x_chunks = int((SIZE_X - CHUNK_SIZE) / DEL_PIX)
        y_chunks = int((SIZE_Y -CHUNK_SIZE) / DEL_PIX)
        n_chunks = x_chunks * y_chunks

        print(f'[INFO] Provided image size : {self.cd.shape}')
        print(f'[INFO] generating segments of size : {CHUNK_SIZE}x{CHUNK_SIZE}')
        print(f'[INFO] Total number of images to be generated ({x_chunks} x {y_chunks}): {n_chunks}')
        N_img = 0
        print('[TASK] Generating only the segments containing a filament..')
        # Given the argument 'chunk_loc' such that this function
        # can be used in the application part, by saving the chunks in
        # a temporary location, where we just 
        # have to chop any arbitrary image such that it canbe 
        # given to the trained model
        if(chunk_loc=='temp'):
            os.system('rm -r temp')
            os.system('mkdir temp')
            os.system('mkdir temp/CD')
            os.system('mkdir temp/skeleton')
        for x in tqdm(range(x_chunks)[:]):
            for y in range(y_chunks)[:]:
                x_start, x_end = int(x * DEL_PIX ) ,  int(x * DEL_PIX + CHUNK_SIZE)
                y_start, y_end = int(y * DEL_PIX)  ,  int(y * DEL_PIX + CHUNK_SIZE)
                if(fil_only):
                    if(len(np.unique(self.fil[x_start:x_end, y_start:y_end]))>1):
                        N_img +=1
                        # self.regional_norm_segment(x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE ,  DEL_PIX)
                        self.local_norm_segment(x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE ,  DEL_PIX)
                else:
                    N_img +=1
                    self.local_norm_segment(x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE,  DEL_PIX)
    
        print(f'[INFO] Total Image generated: {N_img}')

    def get_segment(self,x,y, CHUNK_SIZE, DEL_PIX):
        '''
        return a segment of given CD
        '''
        try:
            x_start, x_end = int(x * DEL_PIX ) ,  int(x * DEL_PIX + CHUNK_SIZE)
            y_start, y_end = int(y * DEL_PIX)  ,  int(y * DEL_PIX + CHUNK_SIZE)
            cd_arr = self.cd[x_start:x_end, y_start:y_end]
            if(cd_arr.shape==(CHUNK_SIZE,CHUNK_SIZE)):
                # print('returning true value')
                return cd_arr
            else:
                # print('returning zero value')
                return np.zeros((CHUNK_SIZE, CHUNK_SIZE))
        except Exception as e:
            print('>>>>', x,y, '---', e)
            return np.zeros((CHUNK_SIZE, CHUNK_SIZE))

    def local_norm_segment(self,x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE,  DEL_PIX):
        # print('[TASK]>> Doing Local Normalisation')
        temp_seg = self.get_segment(x,y, CHUNK_SIZE, DEL_PIX)
        max_im = np.amax(temp_seg)
        if(max_im>0):
            hdu_cd = fits.PrimaryHDU(temp_seg / np.amax(temp_seg))
        else:
            hdu_cd = fits.PrimaryHDU(temp_seg)
        hdul_cd = fits.HDUList([hdu_cd])
        hdul_cd.writeto(f'{chunk_loc}/CD/{self.name}_{x}_{y}.fits', overwrite=True)
        hdu_fil = fits.PrimaryHDU(self.fil[x_start:x_end, y_start:y_end])
        hdul_fil = fits.HDUList([hdu_fil])
        hdul_fil.writeto(f'{chunk_loc}/skeleton/fil_{self.name}_{x}_{y}.fits', overwrite=True)
        hdul_fil.close()


    def regional_norm_segment(self,x,y, x_start, x_end, y_start, y_end, chunk_loc, CHUNK_SIZE,  DEL_PIX):
        '''
        Use the function get_segment to get the surrounding segments of a given area (defined by x,y) and then normalise the central segment by using the maximum values of the surounding region.
        '''
        xl = [x,x-1,x+1]
        yl = [y,y-1,y+1]
        mean = 0
        i = 0
        for xi in xl:
            for yi in yl:
                temp = self.get_segment(xi,yi, CHUNK_SIZE, DEL_PIX)
                i+=1
                mean += np.mean(temp)
                # print(xi,yi ,temp.shape)
                # segment_max = np.mean(temp)
                # if(segment_max>max_cd):
                #     max_cd = segment_max
        mean = mean/i


        hdu_cd = fits.PrimaryHDU(self.get_segment(x,y, CHUNK_SIZE, DEL_PIX) / mean)
        hdul_cd = fits.HDUList([hdu_cd])
        hdul_cd.writeto(f'{chunk_loc}/CD/{self.name}_{x}_{y}.fits', overwrite=True)
        hdu_fil = fits.PrimaryHDU(self.fil[x_start:x_end, y_start:y_end])
        hdul_fil = fits.HDUList([hdu_fil])
        hdul_fil.writeto(f'{chunk_loc}/skeleton/fil_{self.name}_{x}_{y}.fits', overwrite=True)




from tensorflow import keras
import os

def open_fits(path):
    '''
    opens FITS file and 
    '''
    #print('path is :' , path)
    # data = []
    # with open(path, 'b') as f: 
    # f = open(path)
    hdu = fits.open(path)
    data = hdu[0].data
    hdu.close()
    # f.close()

    return data


class filaments(keras.utils.Sequence):


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
        #print(img_paths)
        x_temp, y_temp = [] ,  [] 
        # max_img = 0 # Global maximum of the image, to be used to normalise each image : wrong , better not to normalise image here, shifting the normalisation process to the chopping step.
        for im , tg in zip(img_paths, target_paths):
            # temp_im = np.log10(open_fits(im))
            hdu = fits.open(im)
            data = hdu[0].data
            hdu.close()
            temp_im = np.nan_to_num(data , nan=0, posinf=10, neginf=0)
            hdu = fits.open(tg)
            data = hdu[0].data
            hdu.close()
            temp_tg = data
            temp_tg[temp_tg<0] = 0 
            temp_tg[temp_tg>0] = 1

            x_temp.append((temp_im))
            y_temp.append((temp_tg))
        x_temp = np.asarray(x_temp)
        y_temp = np.asarray(y_temp)
        return x_temp,y_temp

        



# aquila_cd = fits.open('data/arzomanium/HGBS_aquilaM2_column_density_map.fits.gz')
# aquila_fil = fits.open('data/arzomanium/HGBS_aquilaM2_skeleton_map.fits.gz')
# aquila = Cloud('aquila', aquila_cd, aquila_fil)

# # aquila.plot_cloud()
# aquila.create_im_chunks()
