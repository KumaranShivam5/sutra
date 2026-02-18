from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Activation, BatchNormalization



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
