#echo "# autoencorder" >> README.md
#git init
#git add README.md
#git commit -m "first commit"
#git remote add origin https://github.com/Wenlong-Y/autoencorder.git
#git push -u origin master

#setup repository on github

reticulate::use_python("C:\\Users\\ywlon\\AppData\\Local\\r-miniconda", required = TRUE)



#install keras
devtools::install_github("rstudio/keras")
library(keras)
install_keras()

#or

### Run these from RGUi
install.packages("installr")
installr::updateR()

install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)

install.packages("Rcpp")
install.packages("devtools")
devtools::install_github("rstudio/reticulate", force=TRUE)
devtools::install_github("r-lib/debugme")
devtools::install_github("r-lib/processx")
devtools::install_github("tidyverse/rlang")
devtools::install_github("tidyverse/glue")
devtools::install_github("tidyverse/tidyselect")
devtools::install_github("rstudio/tfruns")
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
devtools::install_github("jeroen/jsonlite")



#install H2O
install.packages("h2o")
library(h2o)
h2o.init()



#first example, 2d

library(MASS)
library(keras)
Sigma <- matrix(c(1,0,0,1),2,2)
n_points <- 10000
df <- mvrnorm(n=n_points, rep(0,2), Sigma)
df <- as.data.frame(df)

hist(df$V1,breaks = 20)
boxplot(df)

# Set the outliers
n_outliers <- as.integer(0.01*n_points)
idxs <- sample(n_points,size = n_outliers)
outliers <- mvrnorm(n=n_outliers, rep(5,2), Sigma)
df[idxs,] <- outliers

library(tidyverse)
library(ggplot2)
df %>% mutate(inout = V1 %in% outliers) %>% ggplot() + geom_point(aes(x=V1,y=V2,color=inout), shape=1)


#encoder and decoder
input_layer <- layer_input(shape=c(2))
encoder <- layer_dense(units=1, activation='relu')(input_layer)
decoder <- layer_dense(units=2)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=c('accuracy'))

plot(history)

preds <- autoencoder %>% predict(df)
colnames(preds) <- c("V1", "V2")
preds <- as.data.frame(preds)
