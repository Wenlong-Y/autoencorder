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


#better version

library(keras)
library(tensorflow)
K <- keras::backend()

# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 256L
epochs <- 50L
epsilon_std <- 1.0

# Model definition --------------------------------------------------------

x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss, experimental_run_tf_function = FALSE)
      
 


# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")
y_train <- mnist$train$y
y_test <- mnist$test$y

# Model training ----------------------------------------------------------

history <- vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)
plot(history)

library(tidyverse)

library(ggplot2)
preds <- vae %>% predict(x_test)
error <- rowSums((preds-x_test)**2)
evalerr <- data.frame(error=error, class=as.factor(y_test))

# Reshape original and reconstructed
dim(x_test) <- c(nrow(x_test),28,28)
dim(preds) <- c(nrow(preds),28,28)
image(255*preds[1,,], col=gray.colors(3))
y_test[1]
image(255*x_test[1,,], col=gray.colors(3))


x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()


# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()


x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")
y_train <- mnist$train$y
y_test <- mnist$test$y
evalerr <- data.frame(error=error, class=as.factor(y_test))
evalerr %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_col()


#Outlier Detection in MINST

library(keras)
library(tensorflow)
# Switch to the 1-based indexing from R
options(tensorflow.one_based_extract = FALSE)
K <- keras::backend()
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y
## Exclude "0" from the training set. "0" will be the outlier
outlier_idxs <- which(y_train!=0, arr.ind = T)
X_train <- X_train[outlier_idxs,,]
y_test <- sapply(y_test, function(x){ ifelse(x==0,"outlier","normal")})
# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255


original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 256L
X <- layer_input(shape = c(original_dim))
hidden_state <- layer_dense(X, intermediate_dim, activation = "relu")
z_mean <- layer_dense(hidden_state, latent_dim)
z_log_sigma <- layer_dense(hidden_state, latent_dim)

sample_z<- function(params){
  z_mean <- params[,0:1]
  z_log_sigma <- params[,2:3]
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=1.0
  )
  z_mean + k_exp(z_log_sigma/2)*epsilon
}



z <- layer_concatenate(list(z_mean, z_log_sigma)) %>% 
  layer_lambda(sample_z)
decoder_hidden_state <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
hidden_state_decoded <- decoder_hidden_state(z)
decoded_X_mean <- decoder_mean(hidden_state_decoded)


variational_autoencoder <- keras_model(X, decoded_X_mean)
encoder <- keras_model(X, z_mean)
decoder_input <- layer_input(shape = latent_dim)
decoded_hidden_state_2 <- decoder_hidden_state(decoder_input)
decoded_X_mean_2 <- decoder_mean(decoded_hidden_state_2)
generator <- keras_model(decoder_input, decoded_X_mean_2)

loss_function <- function(X, decoded_X_mean){
  cross_entropy_loss <- (original_dim/1.0)*loss_binary_crossentropy(X, decoded_X_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_sigma - k_square(z_mean) - k_exp(z_log_sigma), axis = -1L)
  cross_entropy_loss + kl_loss
}


variational_autoencoder %>% compile(optimizer = "rmsprop", loss = loss_function, experimental_run_tf_function = FALSE)
history <- variational_autoencoder %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = 50, 
  batch_size = 256, 
  validation_data = list(X_test, X_test)
)
plot(history)


preds <- variational_autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
eval <- data.frame(error=error, class=as.factor(y_test))
library(dplyr)
library(ggplot2)
eval %>% 
  ggplot(aes(x=class,fill=class,y=error))+geom_boxplot()


threshold <- 5
y_preds <- sapply(error, function(x){ifelse(x>threshold,"outlier","normal")})


table(y_preds,y_test)


library(ROCR)
pred <- prediction(error, y_test)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
auc <- unlist(performance(pred, measure = "auc")@y.values)
auc
plot(perf, col=rainbow(10))




#Text fraud detection

# Enron email fraud

#install.packages("tm")
#install.packages("SnowballC")

df <- read.csv("./data/enron.csv")
names(df)

names(df) <-c("emails","responsive")

library(tm)
corpus <- Corpus(VectorSource(df$email))

corpus <- tm_map(corpus,tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus)
dtm <-  removeSparseTerms(dtm, 0.97)
X <- as.data.frame(as.matrix(dtm))
X$responsive <- df$responsive

# Train, test, split
library(caTools)
set.seed(42)
spl <- sample.split(X$responsive, 0.7)
train <- subset(X, spl == TRUE)
test <- subset(X, spl == FALSE)
train <- subset(train, responsive==0)

X_train <- subset(train,select=-responsive)
y_train <- train$responsive
X_test <- subset(test,select=-responsive)
y_test <- test$responsive


library(keras)
input_dim <- ncol(X_train)
inner_layer_dim <- 32
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(input_layer)
decoder <- layer_dense(units=input_dim)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)
autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))


X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 100, batch_size = 32, 
  validation_data = list(X_test, X_test)
)
plot(history)
