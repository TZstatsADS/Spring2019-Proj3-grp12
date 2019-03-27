import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def crop_sub_imgs_fn(x, size=(96,96), is_random=True):
    x = crop(x, wrg=size[1], hrg=size[0], is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x, size=(96,96)):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[size[0], size[1]], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
