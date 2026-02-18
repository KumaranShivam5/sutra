import numpy as np 
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from tensorflow import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from filnet import filnet
from utility import flatten_image
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import argparse 
argP = argparse.ArgumentParser()
argP.add_argument("-i", help="Input file")
argP.add_argument("-o", help="Output file")
argP.add_argument("-f", help="flatten percentage", default=0)
# argP.add_argument("-s", '--subdivide', help="flatten percentage", default=1)
argP.add_argument("-s", '--subdivide', help="flatten percentage", default=1)
# argP.add_argument("-f", help="flatten percentage")
args = argP.parse_args()

# flatten_percent = 0

input_file = args.i 
output_file = args.o 
flatten_percent = int(args.f)
subdivide_needed = bool(args.subdivide)



cd = input_file
# flatten_image = input('flatten_image? Y/N :')
if(flatten_percent>0):
    cd = flatten_image(input_file)
    


# model = keras.models.load_model('trials/u-net-128.model/')
model = keras.models.load_model('trials/u-net-256-bc-v7.model', compile=False)
# model = keras.models.load_model('trials/u-net-128-dice-coef-bc.model', compile=False)
# model = keras.models.load_model('trials/u-net-512-bc.model', compile=False)
pol = filnet(name='polaris' , CD=cd, sigma=1)
if(subdivide_needed):
    pol.identify_fil(model, del_pix=50)
else:
    pol.identify_fil_subdivided(model, del_pix=50)
pol.filament_map.writeto(output_file, overwrite=True)
import os
os.system(f'ds9 {output_file}')

