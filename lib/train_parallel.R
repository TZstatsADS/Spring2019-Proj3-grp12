
### Train a classification model with training features ###



train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  features from LR images 
  ###  -  responses from HR images
  ### Output: 
  ###  -  a list for trained models
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  
  ### the dimension of response arrat is * x 4 x 3, which requires 12 classifiers
  ### this part can be parallelized
  modelList = foreach (i = 1:12,.export = c("gbm.fit","gbm.perf") ) %dopar% {
    
    ## calculate column and channel
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_train[, , c2]
    labMat <- label_train[, c1, c2]
    fit_gbm <- gbm.fit(x=featMat, y=labMat,
                       # here, to reduce computating workload, I choose 100 as number
                       # of iterations. You can choose a larger number if your machine
                       # has very high calculation speed
                       n.trees=100,
                       # for same reason, I set learning rate as 0.1. Although it may
                       # make model less robust, concerning heavy computing task, I
                       # compromised to reality
                       shrinkage = 0.1,
                       distribution="gaussian",
                       interaction.depth=depth, 
                       bag.fraction = 0.5,
                       verbose=FALSE)
    best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = T)
    list(fit=fit_gbm, iter=best_iter)
  }
  
  ### restart cluster to release memory
  if(.Platform$OS.type=="windows"){
    
    stopCluster(cl)
    cl <- makeCluster(cores)
    registerDoParallel(cl, cores=cores)
    }
  return(modelList)
}
