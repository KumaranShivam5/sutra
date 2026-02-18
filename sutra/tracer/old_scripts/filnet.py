import numpy as np 
from matplotlib import pyplot as plt 

from astropy.io import fits

from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from tqdm import tqdm
import os
from multiprocessing.pool import ThreadPool as Pool


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def create_single_cutout(image,position,size, fname=None):
    '''
    Generates a segment of a image using astropy Cutout2D
    image : FITS file with WCS coordinates header
    position : position of cutout initial pixels
    size : size of cutout in pixel coordinates
    fname : filename to save cutout in.
    '''
    # print(position)
    image = image.copy()
    wcs = WCS(image.header)
    ## Inverting x and y position as in fits file axis 1 is vertical and
    # axis 2 is horizontal
    x,y = position 
    position = y,x
    cutout = Cutout2D(image.data , position , size, wcs=wcs)
    # cutout.plot_on_original(color='white')
    image.data = cutout.data
    # create cutout only for the region non nan values
    # if all nan values are idetified return 0 else save cutout 
    # (if fname is supplied) or return the cutout
    if(sum(sum(~np.isnan(image.data)))):
        image.header.update(cutout.wcs.to_header())
        hdul = fits.HDUList([image])
        if(fname):
            hdul.writeto(fname,overwrite=True)
            hdul.close()
        else:
            return hdul
        return 1 # return 1 means that the given cutout have some data and the corresponding operation was don
    else:
        return 0 # return 0 means the given cutout have all nan values and thus the operation was not done


def subdivide_image(cd, CHUNK_SIZE=160, DEL_PIX=100, chunk_loc=''):
    '''
    This function subdivide given image into chunks of size CHUNK_SIZE with overlapping window of DEL_PIX overlap on both the axis,
    by calling the function create_single_cutout with varying position and fname to subdivide image into several cutouts.
    Returns: fname_list: list of file names of the generated cutout.
    '''
    SIZE_X , SIZE_Y = cd.data.shape
    # print(self.cd.shape)
    # print(type(CHUNK_SIZE))
    # If entire image size is smaller than the chunk size, then first pad the image:
    if(SIZE_X<CHUNK_SIZE or SIZE_Y < CHUNK_SIZE):
        m,n = (CHUNK_SIZE+DEL_PIX+10, CHUNK_SIZE+DEL_PIX+10)
        empty = np.zeros((m,n))

        empty[int((m-SIZE_X)/2):int((m+SIZE_X)/2) ,int((n-SIZE_Y)/2):int((n+SIZE_Y)/2)] = cd.data 
        cd.data = np.array(empty, copy=True)
        # DEL_PIX = 0
    SIZE_X , SIZE_Y = cd.data.shape

    x_chunks = int((SIZE_X - CHUNK_SIZE) / DEL_PIX)
    y_chunks = int((SIZE_Y - CHUNK_SIZE) / DEL_PIX)
    n_chunks = x_chunks * y_chunks
    print(f'>>>[INFO] Provided image size : {cd.data.shape}')
    print(f'>>>[TASK] generating segments of size : {CHUNK_SIZE}x{CHUNK_SIZE}')
    print(f'>>>[INFO] Total number of images to be generated ({x_chunks} x {y_chunks}): {n_chunks}')
    N_img = 0
    xy_pos = []
    fname_list = []
    def chunk_creation(x_y): #defining function for using with multiprocessing.pool
        x,y = x_y
        x_start , y_start =  int(x * DEL_PIX ) , int(y * DEL_PIX ) 
        fname = f'temp/CD/{x}_{y}.fits'
        unq_val = len(np.unique(cd.data[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE]))
        #create chunk only where there is data
        if((unq_val)>1):
            cutout_success = create_single_cutout(
                cd, 
                position=(x_start + int(CHUNK_SIZE/2) , y_start+int(CHUNK_SIZE/2)), # adding size/2 because cutout takes the central position
                size = (CHUNK_SIZE, CHUNK_SIZE),
                fname=fname,
                )
            if(cutout_success):
                # keep the filename only if the cutout is success (only if cutout contains data)
                fname_list.append(fname)
        xy_pos.append((x,y))
    if(chunk_loc==''):
        os.system('rm -r temp/')
        os.system('mkdir -p temp')
        os.system('mkdir -p temp/CD')
        
        # for x in tqdm(range(x_chunks)):
        #     for y in range(y_chunks):
        #         chunk_creation((x,y))
        x_y_arr = np.array(np.meshgrid(np.arange(0,x_chunks), np.arange(0,y_chunks))).T.reshape(-1,2)
        pool = Pool(os.cpu_count()-1)
        pool.map(chunk_creation , x_y_arr)
        # for xi in tqdm(x_y_arr):
        #     chunk_creation(xi)
    print(f'>>>[INFO] Total number of images actually generated: {len(fname_list)}')

    return fname_list, xy_pos




from tensorflow import keras
import math

def preprocess_image_segment(img):
    mx = np.nanmax(img)
    if (mx>0):
        img = img/mx 
    return img

def open_fits(fname):
    # print('this func')
    hdu = fits.open(fname,)
    hdu_temp = hdu.copy()
    # hdu.close()
    return hdu_temp[0]


class CloudSegments(keras.utils.Sequence):
    def __init__(self,batch_size, input_img_paths,  ):
        self.input_img_paths = input_img_paths 
        if(batch_size==-1):
            batch_size = len(input_img_paths)
        self.batch_size = batch_size

    def get_header(self, idx):
        '''
        returns the header for the file in idxth index
        This function is to be used to rejoin the image after 
        model prediction
        '''
        return open_fits(self.input_img_paths[idx]).header

    def __len__(self):
        return math.ceil(len(self.input_img_paths) / self.batch_size)

    def __getitem__(self, idx):
        if(idx > len(self)):
            raise ValueError('Index does not exist')
        img_paths = self.input_img_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        x_temp, y_temp = [] ,  [] 
        for im  in img_paths:
            # temp_im = np.log10(open_fits(im))
            temp_im = np.nan_to_num(open_fits(im).data , nan=0, posinf=10, neginf=0)
            temp_im = preprocess_image_segment(temp_im)
            x_temp.append(temp_im)
        x_temp = np.asarray(x_temp)
        return x_temp

def remove_bkg(cd, sigma):
    '''
    Removes background pixels in the data
    '''
    cd_std = np.nanstd(np.asarray(cd.data, dtype=np.float64))
    bkg_map = (cd.data > sigma*cd_std).astype(int)
    cd.data = cd.data*bkg_map
    return cd


class filnet():

    def __init__(self, name, CD, sigma=3):
        if(type(CD)==str):
            self.CD = open_fits(CD)
        else:
            self.CD = CD
        self.name = name 
        # Removing 3 sigma background
        self.CD = remove_bkg(self.CD, sigma)
        # Perform  global normalisation also
        self.CD.data = preprocess_image_segment(self.CD.data)


    def get_CD(self):
        return self.CD

    def get_CD_bkg_map(self):
        
        raise NotImplementedError('Not implemented yet')

    def _subdivide_image(self, chunk_size, del_pix):
        '''
        Declaring this function, such that this subdivide task can be executed directly from the 
        filnet object
        '''
        self.fname_list, self.coutout_loc =  subdivide_image(self.CD, CHUNK_SIZE=chunk_size, DEL_PIX=del_pix,)
    
    def _predict_on_subdivided_images(self, model):
        '''
        TODO: Filtering the filaments identified based on the signam in the original column den
        '''
        return model.predict(self.cloud_segments[0])

    def _create_fits_from_prediction(self):
        '''
        The predictions from the model is in the form of an array
        This function takes all the predictions and converts them into FITS 
        file and stores in a temporary folder for further operation without returning anything.
        otherwise it willbe difficult for RAM to handle these many fits objects. 
        (FITS object's are way bigger than numpy array.)
        '''
        os.system('rm -r temp/predicted')
        os.system('mkdir temp/predicted')
        self.prediction_fname_list = []
        for i in tqdm(range(len(self.fname_list))):
            predicted_img_array_temp = self.predicted_images[i].reshape((self.chunk_size, self.chunk_size))
            header_of_current_segment = self.cloud_segments.get_header(i)
            hdu = fits.PrimaryHDU(predicted_img_array_temp)
            hdu.header = header_of_current_segment
            hdul = fits.HDUList([hdu])
            fname = f'temp/predicted/{self.fname_list[i][8:]}'
            self.prediction_fname_list.append(fname)
            hdul.writeto(fname,overwrite=True)
            hdul.close()

    def identify_fil(self, model, del_pix = 40):
        self.del_pix = del_pix
        self.chunk_size = model.input_shape[1]
        self._subdivide_image(chunk_size=self.chunk_size, del_pix=del_pix)
        print(f'>>>[TASK] Doing filament identification on chunks using the model {model} on {len(self.fname_list)} images')
        self.cloud_segments = CloudSegments(input_img_paths=self.fname_list, batch_size=-1)
        
        self.predicted_images = self._predict_on_subdivided_images(model)
        print(f'>>>[TASK] Creating temporary FITS file from the segmented images')
        self._create_fits_from_prediction()
        print(f'>>>[TASK] rejoining the predicted images')
        self.filament_map = self.rejoin_predictions()
        print('filament map created and can be accessed with `filament_map` attribute')
        return None
        # self.filament = final_img

    def identify_fil_subdivided(self, model, del_pix = 40):
        # self.del_pix = del_pix
        # self.chunk_size = model.input_shape[1]
        # self._subdivide_image(chunk_size=self.chunk_size, del_pix=del_pix)
        # print(f'>>>[TASK] Doing filament identification on chunks using the model {model} on {len(self.fname_list)} images')
        self.cloud_segments = CloudSegments(input_img_paths=self.fname_list, batch_size=-1)
        
        self.predicted_images = self._predict_on_subdivided_images(model)
        print(f'>>>[TASK] Creating temporary FITS file from the segmented images')
        self._create_fits_from_prediction()
        print(f'>>>[TASK] rejoining the predicted images')
        self.filament_map = self.rejoin_predictions()
        print('filament map created and can be accessed with `filament_map` attribute')
        return None

    def rejoin_predictions(self):
        '''
        read the predicted fits file from the temp folder (generated using 
        `create_fits_from_prediction` function)
        and combine using the WCS coordinates with the help of `reproject` module.
        '''
        class hdu_list:
            '''
            to rejoin we need to have an array of all the HDUs at once. 
            Loading all the HDUs in an array will be too much for the ram. 
            Using the __getitem__ function of this class wecan avoid this.
            The hdu object will be returned only when required.
            '''
            def __init__(self, fname_list, chunk_size, del_pix):
                self.chunk_size = chunk_size
                self.fname_list = fname_list
                if(del_pix>40):
                    crop_size = int((chunk_size-del_pix)/2)
                    self.new_size = int(chunk_size - crop_size)
                else:
                    self.new_size = 40
            def __len__(self):
                return len(self.fname_list)
            def __getitem__(self, idx):
                cd = open_fits(self.fname_list[idx])
                position  =  (int(self.chunk_size /2) , int(self.chunk_size/2))
                # print('sent position', position)
                # we can crop the boundary of the predicted images by some fixed pixels
                # before stitching them back. It will remove the higher border intensity 
                # in the predicted image, thus removing the border effect.
                # The size of this crop must be less than the size of DEL_PIX.
                # Might go for half of the DEL_PIX. Better to do this crop operation
                # using astropywcs_out, array, footprint
                cropped_cd = create_single_cutout(
                    cd, 
                    position=position, # adding size/2 because cutout takes the central position
                    size = (self.new_size, self.new_size),
                    )
                return cropped_cd[0]

        hdu_list = hdu_list(self.prediction_fname_list[:], self.chunk_size, self.del_pix)
        from reproject.mosaicking import find_optimal_celestial_wcs
        from reproject import reproject_interp
        from reproject.mosaicking import reproject_and_coadd
        print(">>> [INFO:reproject] : Finding optimal WCS coordinate")
        
        wcs_out , shape_out = find_optimal_celestial_wcs(hdu_list,)

        array, footprint = reproject_and_coadd(
                                input_data=hdu_list,
                                output_projection=wcs_out,
                                shape_out = shape_out , 
                                reproject_function=reproject_interp,
                                match_background = False,
                                parallel=True,
                                combine_function = 'mean'
                            )
        
        # for f in self.fname_list:
        # from astropy.
        array = np.nan_to_num(array)
        skl_hdu = fits.PrimaryHDU(data=array, header=wcs_out.to_header())
        footprint_hdu = fits.ImageHDU(data=footprint, header = wcs_out.to_header())
        hdul = fits.HDUList([skl_hdu, footprint_hdu], )
        return hdul
    
    def cleanup(self):
        # os.system('rm -r tmep')
        return None


    def __getitem__(self, idx):
        if(len(self.fname_list)):
            return fits.open(self.fname_list[idx])
        else:
            raise ValueError('Create image segments first using "segment_image function"')
    
    def test(self):
        return self[0]




from astropy.io import fits



class file_array():
    '''
    Proxy for array of files stored in the filder filepath
    and can be opened with the function file_opener
    '''
    def __init__(self, filepath, file_opener) -> None:
        self.file_names = np.sort([f'{filepath}{i}' for i in os.listdir(filepath)])
        self.file_opener = file_opener
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        return self.file_opener(self.file_names[idx])

class fits_file_array(file_array):
    '''
    Use the class file_array to open list of fits file
    using open_fits function as file_opener function
    '''
    def __init__(self, filepath, file_opener = open_fits) -> None:
        super().__init__(filepath, file_opener)
    
    
    def create_mosaic():
        return None

def fil_to_skl_skimg(fil_array):
    '''
    takes array of fits file of predicted filaments
    then use sklearn to skeletonize
    '''
def fil_to_skl_filfinder(fil_array):
    '''
    use filfinder to skeletonize the 
    '''
    raise NotImplementedError('Not implemented')

def create_mosaic(file_array):
    '''
    can take psudo file array or an actual array
    '''
    hdu_list = [i for i in file_array][:10]
    # return hdu_list

    from reproject.mosaicking import find_optimal_celestial_wcs
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd

    wcs_out , shape_out = find_optimal_celestial_wcs(hdu_list,)

    array, footprint = reproject_and_coadd(
                            input_data=hdu_list,
                            output_projection=wcs_out,
                            shape_out = shape_out , 
                            reproject_function=reproject_interp,
                            match_background = False
                        )
    
    # for f in self.fname_list:
    # from astropy.
    skl_hdu = fits.PrimaryHDU(data=array, header=wcs_out.to_header())
    footprint_hdu = fits.ImageHDU(data=footprint, header = wcs_out.to_header())
    hdul = fits.HDUList([skl_hdu, footprint_hdu], )
    return hdul
