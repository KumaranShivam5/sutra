from keras import backend as K
import numpy as np

import seaborn as sns 
sns.set_style('white')

from keras.models import load_model
from keras import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Activation, BatchNormalization

from tqdm.notebook import tqdm_notebook

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.utils.vis_utils import plot_model



def encoder_block(input_layer, n_filters , maxpool=True , name= ''):
    x = Conv2D(n_filters*1, 3 , activation='relu' , kernel_initializer='HeNormal', padding='same')(input_layer)
    x = Conv2D(n_filters*1, 3 , activation='relu' , kernel_initializer='HeNormal',padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    skip_layer = x 
    if maxpool:
        x = MaxPooling2D(pool_size=(2,2))(x)
    print('_________________________________________\n')
    print(f'{name} : ')
    print(f'>>>>> ', input_layer.shape) # shape of input layer
    print(f'||    ', skip_layer.shape) # Shape of layer going for skip connection
    print(f'<<<<< ', x.shape) # Shape of layer going for output
    return x , skip_layer

def decoder_block(input_layer, skip_layer , n_filters, name=''):
    x = Conv2DTranspose(n_filters , (3,3) , strides=(2,2), padding='same')(input_layer)
    merge = concatenate([skip_layer, x], axis=3)
    x = Conv2D(n_filters*1, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
    x = Conv2D(n_filters*1,3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    print('_________________________________________\n')
    print(f'{name} : ')
    print(f'>>>>> ', input_layer.shape)
    print(f'<<<<< ', x.shape)
    return x 

from keras.layers import Input , Conv2D
from keras import Model


def get_model_from_weights(input_size):
    img_size = (input_size[0], input_size[0])
    n_filters = 4
    n_enc = 5
    encoder_input = Input(shape=(img_size) + (1,))
    enc_arr , skip_arr , dec_arr = [] ,  [] ,  []
    enc , skip = encoder_block(encoder_input , n_filters, name='encoder block 1')
    enc_arr.append(enc)
    skip_arr.append(skip)

    # ENCODER
    for i in range(int(n_enc-1)):
        enc , skip = encoder_block(enc , n_filters*(2**(i+1)), name=f'encoder block {i+2}')
        enc_arr.append(enc)
        skip_arr.append(skip)

    #BOTTLENECK    
    bottle , skip_bottle = encoder_block(enc , n_filters*(2**(n_enc)) , maxpool=False, name='bottleneck block')
    dec = decoder_block(bottle , skip , n_filters*8, name='Decoder 4')

    #DECODER
    for i in range(n_enc-1):
    #     print(n_enc-i-2)
        dec = decoder_block(dec , skip_arr[n_enc-i-2] , n_filters*(2**(n_enc-i)), name=f'Decoder {n_enc-i-1}')

    decoder_output = Conv2D(1,(1,1), padding='same' , activation='sigmoid')(dec)

    model = Model(encoder_input , decoder_output)
    return model




def dice_loss(y_true, y_pred):
    smooth = 1
    num = K.shape(y_pred)[0]
    # y_pred = y_pred[:,:,:,0]
    y_pred_f = K.reshape(y_pred , (K.shape(y_pred)[0],-1))
    y_true_f = K.reshape(y_true , (K.shape(y_true)[0],-1))
    intersection = K.sum((y_true_f*y_pred_f) , axis=1) 
    union = K.sum(y_pred_f, axis=1) + K.sum(y_true_f, axis=1)
    dice_coef = (2.*intersection + smooth) / (union+smooth)
    dice_loss = 1 - (K.sum(dice_coef) / num)
    # dice_loss = 1-dice_coef
    return dice_loss

def dice_metric(y_true, y_pred):
    smooth = 1
    num = K.shape(y_pred)[0]
    # y_pred = y_pred[:,:,:,0]
    y_pred_f = K.reshape(y_pred , (K.shape(y_pred)[0],-1))
    y_true_f = K.reshape(y_true , (K.shape(y_true)[0],-1))
    intersection = K.sum((y_true_f*y_pred_f) , axis=1) 
    union = K.sum(y_pred_f, axis=1) + K.sum(y_true_f, axis=1)
    dice_coef = (2.*intersection + smooth) / (union+smooth)
    
    # dice_loss = 1-dice_coef
    return K.sum(dice_coef)/num

from keras.losses import binary_crossentropy

def bin_dice_loss(lw={}):
    def inner_function(y_true, y_pred):
        b_loss = binary_crossentropy(y_true, y_pred)
        d_loss = dice_loss(y_true, y_pred)
        b_loss = lw['bin']*b_loss 
        d_loss = lw['dice']*d_loss 
        loss = (d_loss*d_loss) / K.sum(list(lw.values()))
        return loss 
    return inner_function

def custom_loss(loss_fn = [], lw=[]):
    '''
    '''
    def inner_function(y_true , y_pred):
        loss = [li(y_true , y_pred)*lwi for li,lwi in zip(loss_fn, K._to_tensor(lw, dtype='float'))] / K.sum(lw) 
        return K._to_tensor(loss , dtype='float') 

    
    return inner_function

# dice_loss(y_test, y_test_pred)

from tensorflow import keras

def plot_prediction(model, x_test_,y_test_):
    # print(x_test_.shape)
    x_test = np.asarray([x_test_[i] for i in [14, 27,  41, 100]])
    y_test = np.asarray([y_test_[i] for i in [14, 27,  41, 100]])
    y_test_pred = model.predict(x_test)
    global current_prediction_save_iter
    np.save(f'intermediate_prediction/{current_prediction_save_iter}.npy' , np.asarray(y_test_pred))
    current_prediction_save_iter+=1

class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test,y_test):
        self.x_test = x_test
        self.y_test = y_test 
    def on_train_batch_end(self, batch, logs=None):
        plot_prediction(self.model,self.x_test, self.y_test)
