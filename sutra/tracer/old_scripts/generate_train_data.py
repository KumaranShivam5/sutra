import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pandas as pd 
import xarray as xr

def gaussian_filter():
    '''
    Function takes an image as np.array and applies gaussian filter
    TODO: define the function
    '''
    return None

class Cloud:
    def __init__(self , name , cd , fil):
        '''
        cd : FITS file containing the cloud column density as PrimaryHDU
        fil : FITS file containing the filament skeleton as PrimaryHDU
        '''
        cd = np.asarray(cd[0].data)
        cd[np.isnan(cd)] = 0
        fil = np.asarray(fil[0].data)
        fil[np.isnan(fil)] = 0
        self.name = name
        self.cd = cd 
        self.fil = fil

    def get_cd(self):
        return self.cd
    def get_fil(self):
        return self.fil
    
    def plot_cloud(self):
        '''
        plots cloud column density and filament skeleton
        TODO: 
            *   plot using WCS coordinates
            *   Use Plotly or Bokeh
            *   Choose suitable cmap to see the filaments 
        '''
        fig , ax = plt.subplots(nrows=1, ncols=2 , figsize=(10,6))
        ax = np.ravel(ax)
        ax[0].imshow(np.log(self.cd), origin='lower' , cmap='inferno_r')
        ax[1].imshow((self.fil), origin='lower' , cmap='magma_r')
        plt.show()

    def create_im_chunks(self , CHUNK_SIZE = 512 ,  DEL_PIX=256):
        SIZE_X , SIZE_Y = self.cd.shape
        print(self.cd.shape)
        print(type(CHUNK_SIZE))
        x_chunks = int((SIZE_X - CHUNK_SIZE) / DEL_PIX)
        y_chunks = int((SIZE_Y -CHUNK_SIZE) / DEL_PIX)
        n_chunks = x_chunks * y_chunks

        print(f'Total number of images to be generated : {n_chunks}')
        N_img = 0
        for x in range(x_chunks)[:]:
            for y in range(y_chunks)[:]:
                x_start, x_end = int(x * DEL_PIX ) ,  int(x * DEL_PIX + CHUNK_SIZE)
                y_start, y_end = int(y * DEL_PIX)  ,  int(y * DEL_PIX + CHUNK_SIZE)
                if(len(np.unique(self.fil[x_start:x_end, y_start:y_end]))>1):
                    N_img +=1
                    hdu_cd = fits.PrimaryHDU(self.cd[x_start:x_end, y_start:y_end])
                    hdul_cd = fits.HDUList([hdu_cd])
                    hdul_cd.writeto(f'train_data/{x}_{y}.fits', overwrite=True)
                    hdu_fil = fits.PrimaryHDU(self.fil[x_start:x_end, y_start:y_end])
                    hdul_fil = fits.HDUList([hdu_fil])
                    hdul_fil.writeto(f'train_data/{self.name}_{x}_{y}_fil.fits', overwrite=True)
        print(f'Total Image generated: {N_img}')


# aquila_cd = fits.open('data/arzomanium/HGBS_aquilaM2_column_density_map.fits.gz')
# aquila_fil = fits.open('data/arzomanium/HGBS_aquilaM2_skeleton_map.fits.gz')
# aquila = Cloud('aquila', aquila_cd, aquila_fil)

# # aquila.plot_cloud()
# aquila.create_im_chunks()
