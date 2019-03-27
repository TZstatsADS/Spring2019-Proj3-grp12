
### Fit the regression model with testing data ###



test <- function(modelList, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model list using training data
  ###  - processed features from testing images 
  ### Output: 
  ###  - training model specification
  
  ### load libraries
  library("gbm")
  
  predArr <- array(NA, c(dim(dat_test)[1], 4, 3))
  
  pred= foreach(i=1:12)%dopar%{
    fit_train <- modelList[[i]]
    ### calculate column and channel
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_test[, , c2]
    ### make predictions
    predict(fit_train$fit, newdata=featMat, 
                    n.trees=fit_train$iter, type="response")
  }
  for (i in 1:12){
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    predArr[, c1, c2] = pred[[i]]
  }
  return(predArr)
}

