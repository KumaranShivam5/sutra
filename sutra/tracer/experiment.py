import numpy as np 
from astropy.io import fits
from matplotlib import pyplot as plt
import tensorflow as tf
import random 
import os 
from tqdm import tqdm

from .utility_v2 import obsFieldHandler
from .utility_v2 import save_object



# seed_value = 0
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)
# session_conf = tf.compat.v1.ConfigProto()
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph() , config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


class dataProcessor:
    def __init__(self , chunk_params, inp_path =  None) -> None:
        self.inp_path = inp_path 
        # self.op_path = op_path 
        self.chunk_params = chunk_params
        self.global_normalizer = []
        self.local_normalizer = []
        self.skeleton_modifier = []

    def add_global_mod(self, norm_func):
        self.global_normalizer.append(norm_func)
    def add_local_mod(self, norm_func):
        self.local_normalizer.append(norm_func)
    def add_target_mod(self, mod_func):
        self.skeleton_modifier.append(mod_func)
    
    def process_data(self , inp_path=None):
        '''
        
        '''
        CHUNK_SIZE , DEL_PIX = self.chunk_params['size']
        # print(self.inp_path)
        if self.inp_path is not None:
            inp_path = self.inp_path
        for pth in inp_path:
        # %matplotlib qt
            # print(pth)
            (cname,opath,cdpth,filpth,) = pth
            cd = fits.open(cdpth)
            fil = fits.open(filpth)
            cloud = obsFieldHandler(cname, cd, fil, global_normalizer = self.global_normalizer , local_normalizer = self.local_normalizer , skeleton_modifier = self.skeleton_modifier,)
            cloud.create_im_chunks(chunk_loc=opath, CHUNK_SIZE = CHUNK_SIZE ,  DEL_PIX = DEL_PIX,)
            cloud.clear_data()
        # aquila.plot_cloud()
    def get_modifiers(self):
        return {
            'global mods': self.global_normalizer,
            'local mods' : self.local_normalizer,
            'target mods' : self.skeleton_modifier
        }





class theModel:
    def __init__(self, inp_train, inp_test, ) -> None:
        self.input_train_dir = inp_train
        self.input_test_dir = inp_test


    def create_model(self, model, optimizer, loss_function, metrics=None):
        self.model = model
        # model.summary()
        model.compile(
            loss = loss_function , 
            optimizer= optimizer,
            metrics = metrics,
            # jit_compile=True
        )
    def train_model(self, callbacks = None, epochs=220):
        if(self.train_data and self.test_data):
            self.history = self.model.fit(
                self.train_data , 
                validation_data = self.test_data , 
                # validation_split = 0.0,
                callbacks = callbacks , 
                epochs= epochs
            )
        else: raise(ValueError('run gen_train_test_data_first'))

    def gen_train_test_data(self, batch_size=32):
        from scripts.utility_v2 import trainDataLoader
        batch_size = 8
        input_img_dir = f'{self.input_train_dir}/CD'
        target_img_dir = f'{self.input_train_dir}/skeleton'
        CD_img_dir_dir = input_img_dir
        skl_img_sir = target_img_dir
        input_img_paths = sorted([
            os.path.join(CD_img_dir_dir , fname)
            for fname in os.listdir(CD_img_dir_dir)
        ])
        target_img_paths = sorted([
            os.path.join(skl_img_sir , fname)
            for fname in os.listdir(skl_img_sir)
        ])
        self.train_data = trainDataLoader(
            batch_size=batch_size,
            input_img_paths=input_img_paths,
            target_img_paths=target_img_paths)

        input_img_dir = f'{self.input_test_dir}/CD'
        target_img_dir = f'{self.input_test_dir}/skeleton'
        CD_img_dir_dir = input_img_dir
        skl_img_sir = target_img_dir
        input_img_paths = sorted([
            os.path.join(CD_img_dir_dir , fname)
            for fname in os.listdir(CD_img_dir_dir)
        ])
        target_img_paths = sorted([
            os.path.join(skl_img_sir , fname)
            for fname in os.listdir(skl_img_sir)
        ])
        self.test_data = trainDataLoader(
            batch_size=-1,
            input_img_paths=input_img_paths,
            target_img_paths=target_img_paths)
            

def predictior(fname, dataprocessor: dataProcessor , model, verbose=0):
    '''
    Function to make prediction on given input image. This function also applies the data preprocessing.

    ...
    Parameters
    ----------
    fname : str
        path of the input FITS file. The FITS file must contain the image as primary HDU

    Returns
    -------
    HDUList : 
        FITS HDUList, PrimaryHDU contains filament probability map, and the ImageHDU is the footprint of the mosaic operation
    '''

    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from reproject.mosaicking import find_optimal_celestial_wcs
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    dp = dataprocessor
    chunk_params = dp.chunk_params['size']
    size = chunk_params[0]
    delpix = chunk_params[1]
    cd = fits.open(fname)[0]
    for fl in dp.global_normalizer:
        cd.data = fl(cd.data)

    nx , ny = int((cd.data.shape[0] - size) / delpix) , int((cd.data.shape[1] - size) / delpix)
    n_chunks = nx * ny
    wcs = WCS(cd.header)
    cutout_wcs_array = []
    cutout_data_array = []
    print('[INFO] >>> Subdividing input image')
    for x in tqdm(range(nx)):
        for y in range(ny):
            x_start, y_start = int(x * delpix), int(y * delpix)
            position = (y_start + int(size/2) , x_start+int(size/2))
            try:
                cutout = Cutout2D(cd.data, position, size, wcs=wcs)
                # # return cutout
                if(len(np.unique(cutout.data))>1):
                    for fl in dp.local_normalizer:
                        cutout.data = fl(cutout.data)
                    cutout_wcs_array.append(cutout.wcs)
                    cutout_data_array.append(cutout.data)
                # # else: print('some error')
                # cutout_data_array.append(cutout)
            except:
                print('Not happening')
    # return cutout_data_array
    print('Size of the chunks : ', size)
    print('total Number of chunks available: ', n_chunks)
    print('total image chunks generated : ', len(cutout_data_array))
    print('[INFO] >>> Generating skeleton probability map')
    # return cutout_data_array
    cutout_pred_array = model.predict(np.array(cutout_data_array)).squeeze()
    del cutout_data_array

    hdu_arr = np.array([fits.PrimaryHDU(data=d, header=h.to_header()) for d,h in zip(cutout_pred_array, cutout_wcs_array)])

    ## removing 20 pixels border form images
    cutout_wcs_array = []
    cutout_pred_array = []
    for h in hdu_arr:
        position = int(size/2) , int(size/2)
        cutout = Cutout2D(h.data, position, int(size-10), wcs=WCS(h.header))
        cutout_wcs_array.append(cutout.wcs)
        cutout_pred_array.append(cutout.data)
    hdu_arr = np.array([fits.PrimaryHDU(data=d, header=h.to_header()) for d,h in zip(cutout_pred_array, cutout_wcs_array)])        


    del cutout_wcs_array
    del cutout_pred_array
    # del cutout_data_array
    print('[INFO] >>> Generating optimal WCS coordinates')
    wcs_out, shape_out = find_optimal_celestial_wcs(hdu_arr)

    print('[INFO] >>> Creating Mosaic')
    array, footprint = reproject_and_coadd(
                                input_data=hdu_arr,
                                output_projection=wcs_out,
                                shape_out = shape_out , 
                                reproject_function=reproject_interp,
                                match_background = False,
                                parallel=False,
                            )
    skl_hdu = fits.PrimaryHDU(data=array, header=wcs_out.to_header())
    footprint_hdu = fits.ImageHDU(data=footprint, header = wcs_out.to_header())
    hdul = fits.HDUList([skl_hdu], )
    return hdul

from .ML_utility import get_model_from_weights

class Experiment():
    '''
    To store all the information of the experimant

    ...

    Attributes
    ----------
    exp_name : str
        name of the experiment
    dataprocessors : dataProcessor
        dataProcesor object used for preprocessing the current experiment data
    model_loc: str
        locatiopn of the model

    Methods
    -------
    get_model() : 
        use keras to load the model
    get_data_mods():
        get the dictionary of functions inside the dataprocessing object
    add_plot(plot, name, save_plot=False):
        add customs plots generated in this experiment in a dictionary format
    add_exp_details(name, value):
        add other experimant details in a dictionary format

    '''

    def __init__(self, exp_name : str, datapreprocessor : dataProcessor , model_loc : str) -> None:
        self.datapreprocessor = datapreprocessor
        self.exp_name = exp_name 
        self.model_loc = model_loc
        self.plots = {}
        self.exp_details = {}

    def __str__(self) -> str:
        print(f'Experiment : {self.exp_name}')
        print(f'Other Experiment Details : {self.exp_details}')
        # Print('')
    
    def get_model(self):
        import keras
        return keras.models.load_model(self.model_loc, compile = False)
    
    def get_model_from_weights(self, weights_loc):
        model = get_model_from_weights(self.datapreprocessor.chunk_params['size'])
        model.load_weights(weights_loc)
        self.model = model
        return model

    def get_data_mods(self):
        '''
        Function returns dictionary of the data modifiers used in the experiment
        '''
        return self.datapreprocessor.get_modifiers
    
    def add_plot(self, plot , name : str , save_plot : bool = False):
        '''
        Function to append plots generated in the current experiment

        Parameters
        ----------
        plot : fig
            matplotlib 'Figure' object fr
        name : str
            name of the plot
        '''
        self.plots[name] = plot
    
    def add_exp_details(self, name : str , value):
        self.exp_details[name] = value
    
    # predict = predictior

    def predict(self, fits_img_loc : str, save_loc : str  = None):
        dp = self.datapreprocessor 
        if self.model is None:
            model = self.get_model_from_weights()
        pred = predictior(fits_img_loc, dp, self.model )
        if(save_loc is None) : 
            return pred
        else: pred.writeto(save_loc, overwrite=True)
    
    def save(self, save_loc = None):
        if(save_loc is None):
            save_object(self , f'{self.exp_name}')
        else:
            print('saving experiment')
            save_object(self , f'{save_loc}')



def clear_dir():
    import os
    os.system(r'rm -r train_data/CD')
    os.system(r'rm -r train_data/skeleton')
    os.system(r'mkdir -p train_data/CD')
    os.system(r'mkdir -p train_data/skeleton')
    os.system(r'rm -r test_data/CD')
    os.system(r'rm -r test_data/skeleton')
    os.system(r'mkdir -p test_data/CD')
    os.system(r'mkdir -p test_data/skeleton')