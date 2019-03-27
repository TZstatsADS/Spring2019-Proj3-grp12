# Project: Can you unscramble a blurry image? 
![image](figs/example.png)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2019

+ Team ## Group 12
+ Team members
	+ Hu, Yiyao yh3076@columbia.edu
	+ Wang, Guanren gw2380@columbia.edu
	+ Yang, Xi xy2378@columbia.edu
	+ Zheng, Fei fz2277@columbia.edu

### **Project summary**:  
+ In this project, we created an AI program that can enhance the resolution of blurry and low-resolution images. We 1) implemented the current practice as the baseline model, 2) implemented an improvement to the current practice, and 3) evaluated the performance gain of your proposed improvement against the baseline. We utilized tensorflow in python as improved model.For baseline model we used `doParallel`,`gbm` and `EBImage` library in R to do super-resolution of the images. We calculated the MSE and PSNR for evaluation purpose. Our model is better than the nerest-neighbor method and bilinear interpolation. Additionally, it is better than bicubic interpolation as well.
+ The MSE and PSNR for baseline model are 0.002776163 and 27.41872 respectively, and the MSE and PSNR for improved model are 0.0094 (because R and Python use different scale for RGB, the best way to compare two models is to use PSNR) and 27.8961 respectively, which is obviously better than the baseline models.

### For grading about `srgan` model
When use [SRGAN.ipynb](https://github.com/TZstatsADS/Spring2019-Proj3-grp12/blob/master/doc/SRGAN.ipynb) to run the `srgan` model, just set corresponding `train set path` and `test set path`. Set `validation=Fasle` to use the all the train data to train. Set `n_epoch_init=10` to initialize the model. This might cost about 30 mins locally for each epoch. Set `n_epoch=20` (no less than 10) to train the total `srgan` model. This might cost about 50 mins locally for each epoch.

### Contribution Statement
*Guanren Wang* is responsible for all implementation of the baseline model (GBM) including: deploying Google Cloud VM for parallel computing, feature extraction, cross validation, model training and testing, super-revolution (generate predicted images), final evaluation.

*Fei Zheng* is responsble for all parts of srgan model except the validation.

*Xi Yang* is responsble for README file and code editing, final test and reproduction of the model.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
