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

+ Project summary: In this project, we created an AI program that can enhance the resolution of blurry and low-resolution images. We 1) implemented the current practice as the baseline model, 2) implemented an improvement to the current practice, and 3) evaluated the performance gain of your proposed improvement against the baseline. We utilized tensorflow in python as improved model.For baseline model we used 'doParallel','gbm' and 'EBImage'library in R to implement super resolution of the images. We calculated the MSE and PSNR for evaluation purpose. Our model is better than the nerest-neighbor method and bilinear interpolation. Additionally, it is better than bicubic interpolation as well.  

+ The MSE and PSNR for R model are 0.002776163 and 27.41872 respectively, and the MSE and PSNR for Python model are 0.0094 (because R and Python use different scale for RGB, the best way to compare two models is to use PSNR) and 27.8961 respectively, which is obviously better than the baseline models.
	

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
