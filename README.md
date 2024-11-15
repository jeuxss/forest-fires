# Algerian Forest Fires 

This project attempts to analyze/perform regression on Algerian forest fire data, taken from [https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset](url). 
Regression methods used include multiple linear, exponential, lasso, and ridge regression, as well as spline regression/additive models.

## Motivation + Methods
The aim of the project was to reinforce/carry out continuous response regression methods.  Given that the original response in the dataset is categorical, significant preprocessing has been done to convert it into a continuous response that goes from 0 to 1, much like a probability function. To accomplish this, I found local neighbourhoods (based on the covariates) of any given observation, and then considered surrounding observations in those neighbourhoods. For each neighbourhood, a probability of how likely a fire is to happen is computed using the number of observations that correspond to a fire versus the total number of observations in our neighbourhood. Then, once a certain number of neighbourhoods have been considered, the average of those probabilities will be associated with that observation as the response, and this process is repeated for each observation. In doing so, our response variable is transformed from binary to continuous. 

The main covariates can be divided into measured and calculated/composed. Measured covariates include temperature, relative humidity, wind speed and rainfall; the calculated/composed covariates are Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC), drought
code (DC), and Buildup Index (BI). All of the built models experimented with a combination of measured and calculated covariates for the effect on predicting the probability.  
The best model, based on sum of squares, ended up being the ridge regression model with all coefficients.

