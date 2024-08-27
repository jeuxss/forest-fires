library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(FNN)
library(mgcv)
#library(glmnet)

set.seed(444)

data <- read.csv("C:\\loo\\4C\\STAT 444\\algforestfires.csv")

# Clean column names by removing leading/trailing spaces
names(data) <- trimws(names(data))

# Convert 'DC' column to numeric, forcing errors to NA
data$DC <- as.numeric(data$DC)

# Checking distribution of rain
#boxcox_rain <- powerTransform(data$Rain)

#data <- data %>% mutate(log_rain = log(data$Rain))

# Remove rows with NA values in 'DC' column
data_cleaned <- data %>% drop_na(DC)
data_cleaned <- data_cleaned %>% drop_na(Classes)
data_cleaned <- data_cleaned[, -which(names(data) == "Rain")]

data_cleaned$month <- factor(data_cleaned$month)
levels(data_cleaned$month)
data_cleaned$Region <- factor(data_cleaned$Region)
levels(data_cleaned$Region)


# Exclude 'Classes' and 'Region' columns for standardization
# covariates <- data_cleaned %>% select(Temperature, RH, Ws, FFMC, DMC, DC, ISI, BUI, FWI, day, month, Region)
covariates <- data_cleaned %>% select(FFMC, DMC, DC, ISI, BUI, FWI)
# covariates$month <- factor(covariates$month)
# levels(covariates$month)
# covariates$Region <- factor(covariates$Region)
# levels(covariates$Region)



# Standardize the covariates
preprocess_params <- preProcess(covariates, method = c("center", "scale"))
standardized_covariates <- predict(preprocess_params, covariates)

df_2 <- data_cleaned
df_2_bs <- data_cleaned



### PREPROCESSING: WEIGHTED PROBABILITIES (BEST)

n_neighbors <- 10

# Calculate the response variable with weighted probabilities using
# Gaussian kernel to weight the neighbors
calculate_response_smooth <- function(standardized_covariates, classes, n_neighbors, sigma) {
  nn <- get.knnx(standardized_covariates, standardized_covariates, k = n_neighbors)
  distances <- nn$nn.dist
  indices <- nn$nn.index
  response_probabilities <- sapply(1:nrow(standardized_covariates), function(i) {
    # Gaussian kernel weights
    weights <- exp(-distances[i, ]^2 / (2 * sigma^2))
    weights <- weights / sum(weights)
    fire_count <- sum(weights * (classes[indices[i, ]] == 0))
    return(fire_count)
  })
  return(response_probabilities)
}

# Experiment with different sigma values
sigma_values <- c(0.5, 0.92, 1, 2)
response_probabilities_list <- lapply(sigma_values, function(sigma) {
  calculate_response_smooth(standardized_covariates, data_cleaned$Classes, n_neighbors = 5, sigma)
})

for (i in seq_along(sigma_values)) {
  df_2[[paste0("Response_sigma_", sigma_values[i])]] <- response_probabilities_list[[i]]
}

# Plot histograms of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  hist(df_2[[paste0("Response_sigma_", sigma_values[i])]], breaks = 20, col = "grey", 
       main = paste("Histogram of Response (sigma =", sigma_values[i], ")"), 
       xlab = "Response")
}

# Plot empirical CDF of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2[[paste0("Response_sigma_", sigma_values[i])]]
  plot(ecdf(response_var), main = paste("Empirical CDF (sigma =", sigma_values[i], ")"), 
       xlab = "Response", ylab = "CDF", verticals = TRUE, do.points = FALSE, col = "blue")
}

###### bootstrapping

# Number of bootstrap samples
n_bootstrap <- 100

# Calculate the response variable with bootstrapping
calculate_response_bootstrap <- function(standardized_covariates, classes, n_neighbors, sigma, n_bootstrap) {
  n <- nrow(standardized_covariates)
  response_matrix <- matrix(0, nrow = n, ncol = n_bootstrap)
  
  for (b in 1:n_bootstrap) {
    # Create a bootstrap sample
    bootstrap_indices <- sample(1:n, replace = TRUE)
    bootstrap_covariates <- standardized_covariates[bootstrap_indices, ]
    bootstrap_classes <- classes[bootstrap_indices]
    
    # Calculate the response for the bootstrap sample
    nn <- get.knnx(bootstrap_covariates, standardized_covariates, k = n_neighbors)
    distances <- nn$nn.dist
    indices <- nn$nn.index
    
    response_probabilities <- sapply(1:n, function(i) {
      weights <- exp(-distances[i, ]^2 / (2 * sigma^2))
      weights <- weights / sum(weights)
      fire_count <- sum(weights * (bootstrap_classes[indices[i, ]] == 0))
      return(fire_count)
    })
    
    response_matrix[, b] <- response_probabilities
  }
  
  # Average the response variables from all bootstrap samples
  final_response <- rowMeans(response_matrix)
  return(final_response)
}


# Different Sigma Values
response_probabilities_list <- lapply(sigma_values, function(sigma) {
  calculate_response_bootstrap(standardized_covariates, df_2_bs$Classes, n_neighbors, sigma, n_bootstrap)
})

# Add the new response variable to the cleaned dataframe for each sigma value
for (i in seq_along(sigma_values)) {
  df_2_bs[[paste0("Response_sigma_", sigma_values[i])]] <- response_probabilities_list[[i]]
}

# Plot 2x2 histograms of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2_bs[[paste0("Response_sigma_", sigma_values[i])]]
  hist(response_var, breaks = 14, col = "grey", 
       main = paste("Histogram of Response (sigma =", sigma_values[i], ")"), 
       xlab = "Response")
}



# Plot 2x2 empirical CDFs of the new response variables for each sigma value
par(mfrow = c(2, 2))  # Create a 2x2 plot layout
for (i in seq_along(sigma_values)) {
  response_var <- df_2_bs[[paste0("Response_sigma_", sigma_values[i])]]
  plot(ecdf(response_var), main = paste("Empirical CDF (sigma =", sigma_values[i], ")"),
       xlab = "Response", ylab = "CDF", verticals = TRUE, do.points = FALSE, col = "blue")
}

df_final <- df_2_bs[which(df_2_bs$Response_sigma_0.92 < 1.0 & df_2_bs$Response_sigma_0.92 > 0.0),]

par(mfrow = c(1, 1))
covs <- df_final[, 4:12]
pairs(covs, main = "Pairs Plot of Weather-based Covariates")

lm1 <- lm(Response_sigma_0.92 ~ Temperature + RH + Ws + FFMC + DMC + DC + ISI + BUI + FWI, data = df_final)
lm2 <- lm(Response_sigma_0.92 ~ FFMC + DMC + DC + ISI + BUI + FWI, data = df_final)

summary(lm1)
summary(lm2)

plot(lm1$residuals)
lm2$residuals

###

## RIDGE/ELASTIC NET
library(glmnet)
library(plotmo)

y <- df_final$Response_sigma_0.92
allx <- data.matrix(covs)
# head(df_final)
mx <- data.matrix(df_final[, 3:12])
cx <- data.matrix(df_final[, 7:12])
# head(cx)


## best model (ridge w/ all coefs)
rallfit <- glmnet(x = allx, y = y, alpha = 0, family = "gaussian")
plot(rallfit, label = TRUE)
cvrall <- cv.glmnet(x = allx, y = y, alpha = 0, family = "gaussian")
plot(cvrall)
plotres(rallfit)
sqrt(min(cvrall$cvm))
cvrall$lambda.min
# cvrall
min(cvrall$cvm)


rmfit <- glmnet(x = cx, y = y, alpha = 0, family = "gaussian")
plot(rmfit, label = TRUE)
cvrm <- cv.glmnet(x = cx, y = y, alpha = 0, family = "gaussian")
plot(cvrm)
plotres(rmfit)
sqrt(min(cvrm$cvm))


lallfitted <- predict(cvlall, allX, s = 'lambda.min')
lallres <- ys1 - lallfitted
plot(lallres, lallfitted)


lmsfit <- glmnet(x = mx, y = ys1, alpha = 1, family = "gaussian")
plot(lmsfit, label = TRUE)
cvlms <- cv.glmnet(x = mX, y = ys1, alpha = 1, family = "gaussian")
plot(cvlms)
plotres(lmsfit)

sqrt(min(cvlms$cvm))

lmsfitted <- predict(cvlms, mX, s = 'lambda.min')
lmsres <- ys1 - lmsfitted
plot(lmsres, lmsfitted)



rmsfit <- glmnet(x = mx, y = ys1, alpha = 0, family = "gaussian")
plot(rmsfit, label = TRUE)
cvrms <- cv.glmnet(x = mX, y = ys1, alpha = 0, family = "gaussian")
plot(cvrms)
plotres(rmsfit)
sqrt(min(cvrms$cvm))

rmsfitted <- predict(cvrms, mX, s = 'lambda.min')
rmsres <- ys1 - rmsfitted
plot(rmsres, rmsfitted)



lclfit <- glmnet(x = cx, y = ys1, alpha = 1, family = "gaussian")
plot(lclfit, label = TRUE)
cvlcl <- cv.glmnet(x = cX, y = ys1, alpha = 1, family = "gaussian")
plot(cvlcl)
plotres(lclfit)

sqrt(min(cvlcl$cvm))

lclfitted <- predict(cvlcl, cX, s = 'lambda.min')
lclres <- ys1 - lclfitted
plot(lclres, lclfitted)


rclfit <- glmnet(x = cx, y = ys1, alpha = 0, family = "gaussian")
plot(rclfit, label= TRUE)
cvrcl <- cv.glmnet(x = cX, y = ys1, alpha = 0, family = "gaussian")
plot(cvrcl)
plotres(rclfit)
sqrt(min(cvrcl$cvm))

## for curiousity in variable selection
lallfit <- glmnet(x = allx, y = y, alpha = 1, family = "gaussian")
plot(lallfit, label = TRUE)

lmfit <- glmnet(x = mx, y = y, alpha = 1, family = "gaussian")
plot(lmfit, label = TRUE)



# elastic net

### elastic net
## all coefs

train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 2,
                              search = "grid",
                              verboseIter = TRUE)


enallfit <- train(allx, y, method = "glmnet", metric = "RMSE",
                  tuneGrid = expand.grid(.alpha = seq(0, 1, length.out = 400),  # optimize an elnet regression
                    .lambda = seq(0, 1, length.out = 400)),
                  trControl = train_control)


enallfit$bestTune
(enallfit$results)[4401,]

# plot(enallfit$bestTune)
## measured (really calculated im just dumb) coefs

enmxfit <- train(mx, y, method = "glmnet", metric = "RMSE",
                  tuneGrid = expand.grid(.alpha = seq(0, 1, length.out = 200),  # optimize an elnet regression
                                         .lambda = seq(0, 1, length.out = 200)),
                  trControl = train_control)
enmxfit$bestTune
enmxfit$results[3001,]
# 
# 
# ### 


## regular splines



### additive spline
knots <- 10

## best model (calced coefs)
mod1 <- gam(y ~ s(FFMC, bs = "bs", k = knots)
            + s(DMC, bs = "bs", k = knots)
            + s(DC, bs = "bs", k = knots)
            + s(ISI, bs = "bs", k = knots)
            + s(BUI, bs = "bs", k = knots)
            + s(FWI, bs = "bs", k = knots), data = df_final)
summary(mod1)
plot(mod1)

par(mfrow=c(1,1))
plot(residuals(mod1) ~ fitted(mod1),
     xlab = "Fitted values", ylab = "Residuals",
     main = "GAM FWI Covs residuals")
# curve <- smooth(residuals(mod1) ~ fitted(mod1))
qqnorm(residuals(mod1))

## all coefs
mod2 <- gam(y ~ s(Temperature, bs = "bs", k = knots)
            + s(Ws, bs = "bs", k = knots)
            + s(RH, bs = "bs", k = knots)
            + s(FFMC, bs = "bs", k = knots)
            + s(DMC, bs = "bs", k = knots)
            + s(DC, bs = "bs", k = knots)
            + s(ISI, bs = "bs", k = knots)
            + s(BUI, bs = "bs", k = knots)
            + s(FWI, bs = "bs", k = knots),
            data = df_final)



summary(mod2)


par(mfrow=c(1,2))
plot(residuals(mod2) ~ fitted(mod2),
     xlab = "Fitted values", ylab = "Residuals")
qqnorm(residuals(mod2))


mod3 <- gam(y ~ s(Temperature, bs = "bs", k = knots)
            + s(RH, bs = "bs", k = knots)
            + s(Ws, bs = "bs", k = knots), data = df_final)
summary(mod3)
plot(mod3)

par(mfrow=c(1,2))
plot(residuals(mod3) ~ fitted(mod3),
     xlab = "Fitted values", ylab = "Residuals")
qqnorm(residuals(mod3))


## polynomial regression

n <- 2
polymodn <- lm(y ~ poly(cx, degree = n, raw = TRUE))
# xx <- seq(xmin, xmax, length.out = 1e03)
summary(polymodn)
plot(polymodn)


