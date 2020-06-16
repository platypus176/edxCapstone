################################
# Create edx set, validation set
################################
# Install packages if necessary, and load them
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Download  and create database
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove from memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Create test set and train set
################################
# Create test set using 20% of edx data and train set with the rest
set.seed(2, sample.kind = "Rounding")
test_index<-createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set<- edx[test_index,]
train_set<-edx[-test_index,]

# Ensure test set and train set has same users and movies to avoid getting NA
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

################################
# Average model
################################
# Overall average rating of training set
mu<-mean(train_set$rating)

#RMSE for average
RMSE(test_set$rating, mu)

################################
# Movie Effect
################################
# Add movie bias term
b_i<-train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating-mu))

# Predict rating with mean and movie bias term
predicted_ratings<-test_set %>% 
  left_join(b_i, by= 'movieId') %>% 
  mutate(pred=mu+b_i)%>% 
  pull(pred)

# RMSE for movie effect
RMSE(predicted_ratings, test_set$rating)

################################
# Movie and User Effect
################################
# Add user bias term
b_u<- train_set %>% 
  left_join(b_i, by='movieId') %>% 
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict rating with mean, movie and user bias term
predicted_ratings<- test_set %>% 
  left_join(b_i, by= 'movieId') %>% 
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

# RMSE for movie and user effect
RMSE(predicted_ratings, test_set$rating)

########################################
#Regularization of Movie and User Effect
########################################
# Figuring out best lambda
lambdas<- seq(0,8,0.25)

# Regularization function
regularization<-function(x){
  # Overall average rating of train dataset
  mu<-mean(train_set$rating)
  
  # Regularized movie bias term
  b_i<- train_set %>% 
    group_by(movieId)%>% 
    summarize(b_i = sum(rating-mu)/(n()+x))
  
  # Regularized user bias term
  b_u<- train_set %>% 
    left_join(b_i, by= 'movieId') %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i-mu)/(n()+x))
  
  # Predict rating with mean, movie and user bias term
  predicted_ratings<- test_set %>% 
    left_join(b_i, by= 'movieId') %>% 
    left_join(b_u, by= 'userId')%>%
    mutate(pred= mu + b_i + b_u) %>% 
    .$pred
  # RMSE for regularized movie and user effect
  return(RMSE(predicted_ratings, test_set$rating))}

# Obtain RMSE of lambda using regularization function
rmse<- sapply(lambdas, regularization)

# Plot RMSE vs. lambdas 
plot(lambdas, rmse)

# Figuring out the minimized lambda
lambda<-lambdas[which.min(rmse)]
lambda

# Output RMSE of regularized model
regularization(lambda)

################################################################
#Final Modal of Regularization of Movie and User Effect
################################################################
# Figuring out best lambda
lambdas<- seq(0,8,0.25)

# Regularization function
regularization<-function(x){
  # Overall average rating of edx dataset
  mu<- mean(edx$rating)
  
  # Regularized movie bias term
  b_i<- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+x))
  
  # Regularized user bias term
  b_u<- edx %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+x))
  
  # Predictions on validation set using mean and regularized movie and user bias term
  predicted_ratings<- validation %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by="userId") %>%
    mutate(pred= mu +b_i+b_u) %>% 
    pull(pred)
  
  # Final model output RMSE
  return(RMSE(predicted_ratings, validation$rating))}

# Obtain RMSE of lambda using regularization function
rmse<- sapply(lambdas, regularization)

# Plot RMSE vs. lambdas 
plot(lambdas, rmse)

# Figuring out the minimized lambda
lambda<-lambdas[which.min(rmse)]
lambda

# Final model output RMSE of validation set
regularization(lambda)
