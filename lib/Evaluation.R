
### Calculate mse and psnr ###



msepsnr <- function(train_dir, test_dir){
  library("EBImage")

  
  n_files=length(list.files(test_dir))
  for(i in 1:n_files){
    
    imgH <- as.array(readImage(paste0(train_dir,  "img_", sprintf("%04d", i), ".jpg"),type="jpeg"))
    imgP <- as.array(readImage(paste0(test_dir,  "img_", sprintf("%04d", i), ".png"),type="png"))
    
    mse <- mean(abs(imgP-imgH)^2)
    psnr <- 10*log10(1/mse)
    

  }
  mp <- c(mse/n_files,psnr/n_files)
  return(mp)
}

