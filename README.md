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

+ Project summary: In this project, we create an mobile AI program that can enhance the resolution of blurry and low-resolution images. We 1) implement the current practice as the baseline model, 2) implement an improvement to the current practice, and 3) evaluate the performance gain of your proposed improvement against the baseline. We utilize tensorflow in python and 'gbm' library in R to implement super resolution of the images. We calculate the mse and psnr for evaluation purpose. Our model is better than the nerest-neighbor method and bilinear interpolation. Additionally, it is better than bicubic interpolation as well.  

+ The mse and psnr for R model are 0.002776163 and 27.41872 respectively, and the mse and psnr for Python model are 0.0094 and 27.8961 respectively, obviously better than the baseline models.
	

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
