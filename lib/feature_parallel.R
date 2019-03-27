
### Construct features and responses for training images###



feature <- function(LR_dir, HR_dir, n_points=1000){
  
  ### Construct process features for training images (LR/HR pairs)
  
  ### Input: +a path for low-resolution images 
  ###        + a path for high-resolution images 
  ###        + number of points sampled from each LR image
  ### Output: 
  ###        + an .RData file contains processed features and responses for the images
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(LR_dir))

  ### read LR/HR image pairs
  library(abind)
  all_feature=foreach(i = 1:n_files,.export = c("readImage","abind"))%dopar%{
    ### store feature and responses
    
    
    featMat <- array(NA, c(n_points, 8, 3))
    labMat <- array(NA, c(n_points, 4, 3))
    
   
    imgLR <- readImage(paste0(LR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgHR <- readImage(paste0(HR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgLR <- as.array(imgLR)
    imgHR = as.array(imgHR)
    
    ### step 1. sample n_points from imgLR
    dimLR=dim(imgLR)
    select=sample(dimLR[1]*dimLR[2],n_points)
    select_row=(select-1)%%dimLR[1]+1
    select_col=(select-1)%/%dimLR[1]+1
    
    ### step 2. for each sampled point in imgLR,
        ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat by padding
        ### zeros for boundary points
    for(j in 1:3){
        pad=cbind(0,imgLR[,,j],0)
        pad=rbind(0,pad,0)
        center=pad[cbind(select_row+1,select_col+1)]
        featMat[,1,j]=pad[cbind(select_row,select_col)]-center
        featMat[,2,j]=pad[cbind(select_row,select_col+1)]-center
        featMat[,3,j]=pad[cbind(select_row,select_col+2)]-center
        featMat[,4,j]=pad[cbind(select_row+1,select_col+2)]-center
        featMat[,5,j]=pad[cbind(select_row+2,select_col+2)]-center
        featMat[,6,j]=pad[cbind(select_row+2,select_col+1)]-center
        featMat[,7,j]=pad[cbind(select_row+2,select_col)]-center
        featMat[,8,j]=pad[cbind(select_row+1,select_col)]-center
        
        ### step 2.2. save the corresponding 4 sub-pixels of imgHR in labMat
        channelHR=imgHR[,,j]
        
        labMat[,1,j]=channelHR[cbind(select_row*2-1,select_col*2-1)]-center
        labMat[,2,j]=channelHR[cbind(select_row*2-1,select_col*2)]-center
        labMat[,3,j]=channelHR[cbind(select_row*2,select_col*2)]-center
        labMat[,4,j]=channelHR[cbind(select_row*2,select_col*2-1)]-center
        
    ### step 3. repeat above for three channels
    }
    abind(featMat,labMat, along=2)
  }
  all_feature=abind(all_feature,along=1)
  attributes(all_feature)=list(dim=dim(all_feature))
  featMat=all_feature[,1:8,]
  labMat=all_feature[,9:12,]
  
  return(list(feature = featMat, label = labMat))
}
