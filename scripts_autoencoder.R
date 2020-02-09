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

# Coerce the dataframe to matrix to perform the training
df <- as.matrix(df)
history <- autoencoder %>% fit(
  df,df,
  epochs = 100, batch_size = 128,
  validation_split = 0.2
)

plot(history)

preds <- autoencoder %>% predict(df)
colnames(preds) <- c("V1", "V2")
preds <- as.data.frame(preds)

# Coerce back the matrix to data frame to use ggplot later
df <- as.data.frame(df)
# Euclidean distance larger than 3 = sum of squares larger than 9
df$color <- ifelse((df$V1-preds$V1)**2+(df$V2-preds$V2)**2>9,"red","blue")

library(ggplot2)
df %>% ggplot(aes(V1,V2),col=df$color)+geom_point(color = df$color, position="jitter")



#MNIST example

library(keras)
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y

image(X_train[1,,], col=gray.colors(3))
y_train[1]


# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255

input_dim <- 28*28 #784
inner_layer_dim <- 32
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))
history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 50, batch_size = 256, 
  validation_split=0.2
)

plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
error


# Which were more problematic to reconstruct?
eval <- data.frame(error=error, class=as.factor(y_test))
library(dplyr)
eval %>% group_by(class) %>% summarise(avg_error=mean(error))

library(ggplot2)
eval %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_col()

image(255*preds[1,,], col=gray.colors(3))

y_test[1]
image(255*X_test[1,,], col=gray.colors(3))

#Outlier detection in MNIST

library(keras)
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y

## Exclude "7" from the training set. "7" will be the outlier
outlier_idxs <- which(y_train!=7, arr.ind = T)
X_train <- X_train[outlier_idxs,,]
y_test <- sapply(y_test, function(x){ ifelse(x==7,"outlier","normal")})

# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255
input_dim <- 28*28 #784
inner_layer_dim <- 32
# Create the autoencoder
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)
autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))
history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 50, batch_size = 256, 
  validation_split=0.2
)
plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
eval <- data.frame(error=error, class=as.factor(y_test))
library(ggplot2)
library(dplyr)
eval %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_boxplot()

threshold <- 15
y_preds <- sapply(error, function(x) ifelse(x>threshold,"outlier","normal"))



#credit card fraud detection

df <- read.csv("./data/creditcard.csv", stringsAsFactors = F)
head(df)

#sanity check 
library(ggplot2)
library(dplyr)
df %>% ggplot(aes(Time,Amount))+geom_point()+facet_grid(Class~.)

#use keras

# Remove the time and class column
idxs <- sample(nrow(df), size=0.1*nrow(df))
train <- df[-idxs,]
test <- df[idxs,]
y_train <- train$Class
y_test <- test$Class
X_train <- train %>% select(-one_of(c("Time","Class")))
X_test <- test %>% select(-one_of(c("Time","Class")))
# Coerce the data frame to matrix to perform the training
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

#two layer encoders

library(keras)
input_dim <- 29
outer_layer_dim <- 14
inner_layer_dim <- 7
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=outer_layer_dim, activation='relu')(input_layer)
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(encoder)
decoder <- layer_dense(units=inner_layer_dim)(encoder)
decoder <- layer_dense(units=outer_layer_dim)(decoder)
decoder <- layer_dense(units=input_dim)(decoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

summary(autoencoder)
#somehow the book has one more inner layer 

autoencoder %>% compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=c('accuracy'))

history <- autoencoder %>% fit(
  X_train,X_train,
  epochs = 10, batch_size = 32,
  validation_split=0.2
)

plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
preds <- as.data.frame(preds)

y_preds <- ifelse(rowSums((preds-X_test)**2)/30<1,rowSums((preds-X_test)**2)/30,1)

library(ROCR)
pred <- prediction(y_preds, y_test)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

#using H2O to discover credit fraud

install.packages("h2o")
library(h2o)

h2o.init()

df <- read.csv("./data/creditcard.csv", stringsAsFactors = F)
df <- as.h2o(df)


#or df2 <- h2o.uploadFile("./data/creditcard.csv")

splits <- h2o.splitFrame(df, ratios=c(0.8), seed=1)
train <- splits[[1]]
test <- splits[[2]]

label <- "Class" 
features <- setdiff(colnames(train), label)

autoencoder <- h2o.deeplearning(x=features, training_frame = train, autoencoder = TRUE, seed = 1, hidden=c(10,2,10), epochs = 10, activation = "Tanh")
preds <- h2o.predict(autoencoder, test)

# Use the predict function as before
preds <- h2o.predict(autoencoder, test)

head(preds)

library(tidyverse)
anomaly <- h2o.anomaly(autoencoder, test) %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  mutate(Class = as.vector(test[, 31]))


#Image reconstruction using VAEs
library(tensorflow)
library(keras)
# Switch to the 1-based indexing from R
options(tensorflow.one_based_extract = FALSE)
K <- keras::backend()
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y
# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255


orig_dim <- 784
latent_dim <- 2
inner_dim <- 256
X <- layer_input(shape = c(orig_dim))
hidden_state <- layer_dense(X, inner_dim, activation = "relu")
z_mean <- layer_dense(hidden_state, latent_dim)
z_log_sigma <- layer_dense(hidden_state, latent_dim)

sample_z<- function(params){
  z_mean <- params[,1:2]
  z_log_sigma <- params[,3:4]
  epsilon <- K$random_normal(
    shape = c(K$shape(z_mean)[[1]]), 
    mean=0.,
    stddev=1
  )
  z_mean + K$exp(z_log_sigma/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_sigma)) %>%
  layer_lambda(sample_z)

decoder_hidden_state <- layer_dense(units = inner_dim, activation = "relu")
decoder_mean <- layer_dense(units = orig_dim, activation = "sigmoid")
hidden_state_decoded <- decoder_hidden_state(z)
X_decoded_mean <- decoder_mean(hidden_state_decoded)

# end-to-end autoencoder
variational_autoencoder <- keras_model(X, X_decoded_mean)

encoder <- keras_model(X, z_mean)
decoder_input <- layer_input(shape = latent_dim)

loss_function <- function(X, decoded_X_mean){
  cross_entropy_loss <- loss_binary_crossentropy(X, decoded_X_mean)
  kl_loss <- -0.5*K$mean(1 + z_log_sigma - K$square(z_mean) - K$exp(z_log_sigma), axis = -1L)
  cross_entropy_loss + kl_loss
}


variational_autoencoder %>% compile(optimizer = "rmsprop", loss = loss_function, experimental_run_tf_function=FALSE)
history <- variational_autoencoder %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = 10, 
  batch_size = 256, 
  validation_data = list(X_test, X_test)
)
plot(history)

library(tidyverse)
library(ggplot2)
preds <- variational_autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
eval <- data.frame(error=error, class=as.factor(y_test))
eval %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_col()


# Reshape original and reconstructed
dim(X_test) <- c(nrow(X_test),28,28)
dim(preds) <- c(nrow(preds),28,28)
image(255*preds[1,,], col=gray.colors(3))
y_test[1]
image(255*X_test[1,,], col=gray.colors(3))

grid_x <- seq(-4, 4, length.out = 3)
grid_y <- seq(-4, 4, length.out = 3)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, preds[sample(dim(X_test)[1],1),,] %>% matrix(ncol = 28) ) 
  }
  rows <- cbind(rows,column) #this part is omitted in the book
}
rows %>% as.raster() %>% plot()

