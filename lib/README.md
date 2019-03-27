# Project: Can you unscramble a blurry image? 

### Libraries folder
This folder contains all the functions used in *main.rmd*

## SRGAN
The [SRGAN](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/tree/master/lib/SRGAN) folder contains four file which defines the model.

[main.py](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/blob/master/lib/SRGAN/main.py) defines the `train` and `predict` function.

[model.py](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/blob/master/lib/SRGAN/model.py) defines the `generater`, `discriminator` and `vgg19` model.

[utils.py](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/blob/master/lib/SRGAN/utils.py) defines `cropping` and `downsampling` function.

[vgg19.npy](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/blob/master/lib/SRGAN/vgg19.npy) is the pre-trained `VGG19` model.