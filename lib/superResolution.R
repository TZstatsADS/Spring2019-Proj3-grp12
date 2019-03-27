
### Super-resolution ###



superResolution <- function(LR_dir, HR_dir, modelList){
  
  ### Construct high-resolution images from low-resolution images with trained predictor
  
  ### Input: + a path for low-resolution images 
  ###        + a path for high-resolution images 
  ###        + a list for predictors
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(LR_dir))
  
  ### read LR/HR image pairs
  for(i in 1:n_files){
    imgLR <- readImage(paste0(LR_dir,  "img", "_", sprintf("%04d", i), ".jpg"))
    pathHR <- paste0(HR_dir,  "img", "_", sprintf("%04d", i), ".png")
    n<-dim(imgLR)[1]
    m<-dim(imgLR)[2]
    featMat <- array(NA, c(n*m, 8, 3))
    center=featMat
    
    imgLR <- as.array(imgLR)
    select=c(1:(n*m))
    #n_points=length(select)
    select_row=(select-1)%%n +1
    select_col=(select-1)%/%n +1
    
    ### step 1. for each pixel and each channel in imgLR:
    ###           save (the neighbor 8 pixels - central pixel) in featMat by padding
    ###           zeros for boundary points
    for (j in 1:3){
      pad=cbind(0,imgLR[,,j],0)
      pad=rbind(0,pad,0)
      
      featMat[,1,j]=pad[cbind(select_row,select_col)] 
      featMat[,2,j]=pad[cbind(select_row,select_col+1)]
      featMat[,3,j]=pad[cbind(select_row,select_col+2)]
      featMat[,4,j]=pad[cbind(select_row+1,select_col+2)]
      featMat[,5,j]=pad[cbind(select_row+2,select_col+2)]
      featMat[,6,j]=pad[cbind(select_row+2,select_col+1)]
      featMat[,7,j]=pad[cbind(select_row+2,select_col)]
      featMat[,8,j]=pad[cbind(select_row+1,select_col)]
      ### - center
      center[,,j]=imgLR[,,j]
      featMat[,,j]=featMat[,,j]-center[,,j]
    }
    
    ### step 2. apply the modelList over featMat
    predMat <- test(modelList, featMat)
    center=predMat
    ### + center
    for(j in 1:3){
      center[,,j]=imgLR[,,j]
      predMat[,,j]=predMat[,,j]+center[,,j]
    }
    
    ### step 3. recover high-resolution from predMat and save in HR_dir
    if (dim(predMat)[1]== m*n){
      index<-c(1:(m*n))
      index2<-1:m
      featImg<- array(NA, c(2*n, 2*m, 3))
      for (k in 1:3){
        
        vec_odd<-numeric(2*m*n)
        vec_odd[c(2*index-1)]<-as.vector(predMat[,1,k])
        vec_odd[2*index]<-as.vector(predMat[,4,k])
        vec_even<-numeric(2*m*n)
        vec_even[c(2*index-1)]<-as.vector(predMat[,2,k])
        vec_even[2*index]<-as.vector(predMat[,3,k])
        mat_odd<-matrix(vec_odd,nrow = 2*n,ncol = m)
        mat_even<-matrix(vec_even,nrow = 2*n,ncol = m)
        
        imgMAt<-matrix(NA,nrow = 2*n,ncol = 2*m)
        
        imgMAt[,c(2*index2-1)]<-mat_odd
        imgMAt[,2*index2]<-mat_even
        
        featImg[,,k]<-imgMAt
    }}
    
    HRimg<-Image(featImg,colormode = Color)
    writeImage(HRimg, file=pathHR,type = "jpeg")
    
  }
}