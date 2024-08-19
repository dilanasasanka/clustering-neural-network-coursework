# Install required packages
install.packages("readxl")
install.packages("neuralnet")
install.packages("dplyr")
install.packages("Metrics")
install.packages("MLmetrics")

# Load necessary libraries
library(readxl)
library(neuralnet)
library(dplyr)
library(Metrics)
library(MLmetrics)
library(ggplot2)

# Load data
exchange_data <- read_excel("D:/ML/ExchangeUSD.xlsx")

# Extract the third column
exchange_rates <- exchange_data$`USD/EUR`
exchange_rates_df <- data.frame(exchange_rates)

# Split data into training and testing sets
train_data <- exchange_rates_df[1:400,]
test_data <- exchange_rates_df[401:length(exchange_rates),]

# Creating lagged input-output matrix for training data set t4
train_lagged_t4 <- bind_cols(G_prev3 = lag(train_data,4),
                             G_prev2 = lag(train_data,3),
                             G_prev = lag(train_data,2),
                             G_curr = lag(train_data,1),
                             G_pred = train_data)

# Creating lagged input-output matrix for testing data set t4
test_lagged_t4 <- bind_cols(G_prev3 = lag(test_data,4),
                            G_prev2 = lag(test_data,3),
                            G_prev = lag(test_data,2),
                            G_curr = lag(test_data,1),
                            G_pred = test_data)

# Creating lagged input-output matrix for training data set t3
train_lagged_t3 <- bind_cols(G_prev2 = lag(train_data,3),
                             G_prev = lag(train_data,2),
                             G_curr = lag(train_data,1),
                             G_pred = train_data)

# Creating lagged input-output matrix for testing data set t3
test_lagged_t3 <- bind_cols(G_prev2 = lag(test_data,3),
                            G_prev = lag(test_data,2),
                            G_curr = lag(test_data,1),
                            G_pred = test_data)

# Creating lagged input-output matrix for training data set t2
train_lagged_t2 <- bind_cols(G_prev = lag(train_data,2),
                             G_curr = lag(train_data,1),
                             G_pred = train_data)

# Creating lagged input-output matrix for testing data set t2
test_lagged_t2 <- bind_cols(G_prev = lag(test_data,2),
                            G_curr = lag(test_data,1),
                            G_pred = test_data)

# Creating lagged input-output matrix for training data set t1
train_lagged_t1 <- bind_cols(G_curr = lag(train_data,1),
                             G_pred = train_data)

# Creating lagged input-output matrix for testing data set t1
test_lagged_t1 <- bind_cols(G_curr = lag(test_data,1),
                            G_pred = test_data)

# Display the input-output matrices as tables
View(train_lagged_t4)
View(test_lagged_t4)

# Remove N/A values from t4
train_lagged_t4 <- train_lagged_t4[complete.cases(train_lagged_t4),]
test_lagged_t4 <- test_lagged_t4[complete.cases(test_lagged_t4),]

# Display the cleaned input-output matrices as tables
View(train_lagged_t4)
View(test_lagged_t4)

# Remove N/A values from t3
train_lagged_t3 <- train_lagged_t3[complete.cases(train_lagged_t3),]
test_lagged_t3 <- test_lagged_t3[complete.cases(test_lagged_t3),]

# Remove N/A values from t2
train_lagged_t2 <- train_lagged_t2[complete.cases(train_lagged_t2),]
test_lagged_t2 <- test_lagged_t2[complete.cases(test_lagged_t2),]

# Remove N/A values from t1
train_lagged_t1 <- train_lagged_t1[complete.cases(train_lagged_t1),]
test_lagged_t1 <- test_lagged_t1[complete.cases(test_lagged_t1),]

# Function to normalize data
normalize_data <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize data for t4
train_normalized_t4 <- as.data.frame(lapply(train_lagged_t4, normalize_data))
test_normalized_t4 <- as.data.frame(lapply(test_lagged_t4, normalize_data))

# Normalize data for t3
train_normalized_t3 <- as.data.frame(lapply(train_lagged_t3, normalize_data))
test_normalized_t3 <- as.data.frame(lapply(test_lagged_t3, normalize_data))

# Normalize data for t2
train_normalized_t2 <- as.data.frame(lapply(train_lagged_t2, normalize_data))
test_normalized_t2 <- as.data.frame(lapply(test_lagged_t2, normalize_data))

# Normalize data for t1
train_normalized_t1 <- as.data.frame(lapply(train_lagged_t1, normalize_data))
test_normalized_t1 <- as.data.frame(lapply(test_lagged_t1, normalize_data))

# Plot boxplot for normalized training data and original(t4)
boxplot(train_normalized_t4)
boxplot(train_lagged_t4)

# Get original data from denormalized data
train_original_pred <- train_lagged_t4["G_pred"]
test_original_pred_t1 <- test_lagged_t1["G_pred"]
test_original_pred_t2 <- test_lagged_t2["G_pred"]
test_original_pred_t3 <- test_lagged_t3["G_pred"]
test_original_pred_t4 <- test_lagged_t4["G_pred"]

View(test_original_pred)

# Find min and max value from dataset
min_val <- min(train_original_pred)
max_val <- max(train_original_pred)

# Create function to denormalize data
denormalize_data <- function(x) { 
  min_val <- min(train_original_pred)
  max_val <- max(train_original_pred)
  return( (max_val - min_val) * x + min_val )
}

# train and test neural network function
train_test_neural_net <- function(training_data, testing_data, neurons_hidden_layer, linear_output, activation_function, test_original_pred) {
  
  set.seed(123)
  
  formula_names <- as.formula(paste("G_pred~", paste(names(training_data)[-ncol(training_data)], collapse = "+")))
  
  #Training model
  if (is.null(activation_function)) {
    rates_load_nn <- neuralnet(formula_names, 
                               hidden=neurons_hidden_layer,
                               data=training_data,
                               linear.output=linear_output)
  } else {
    rates_load_nn <- neuralnet(formula_names, 
                               hidden=neurons_hidden_layer,
                               data=training_data,
                               act.fct = activation_function,
                               linear.output=linear_output)
  }
  plot(rates_load_nn)
  
  # Extract column names for testing
  predict_label_removed <- setdiff(names(training_data), "G_pred")
  
  # Testing model
  
  # Removing Prediction variable
  temp_test <- subset(testing_data, select=predict_label_removed)
  head(temp_test)
  
  # Creating predicted variables
  model_results <- compute(rates_load_nn, temp_test)
  predicted_results_norm <- model_results$net.result
  #View(predicted_results_norm)
  
  # unnormalize NN predicted data
  pred_NN_Results <- denormalize_data(predicted_results_norm)
  #View(pred_NN_Results)
  
  par(mfrow = c(1, 2))  # Splitting the plotting area into 1 row and 2 columns
  plot(rates_load_nn)
  # Plotting the scatter plot
  plot(test_original_pred$G_pred, pred_NN_Results, 
       xlab = "Desired Output", ylab = "Predicted Output",
       main = "Comparison of Desired Output vs. Predicted Output")
  
  # Adding a trendline
  abline(lm(pred_NN_Results ~ test_original_pred$G_pred), col = "red")
  
  # Adding a legend
  legend("topleft", legend = "Trendline", col = "red", lty = 1)
  
  # testing performance of RMSE
  rmse <- rmse(exp(pred_NN_Results),test_original_pred$G_pred)
  
  # testing performance of MAE
  mae <- mae(exp(pred_NN_Results),test_original_pred$G_pred)
  
  # testing performance of MAPE
  mape <- mean(abs((test_original_pred$G_pred - pred_NN_Results)/test_original_pred$G_pred))*100
  
  # testing performance of sMAPE
  smape <- mean(2 * abs(test_original_pred$G_pred - pred_NN_Results) / (abs(test_original_pred$G_pred) + abs(pred_NN_Results))) * 100
  
  
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Train and test neural network models with different input vectors and structures
results_1 <- train_test_neural_net(train_normalized_t1, test_normalized_t1, c(4), TRUE, NULL, test_original_pred_t1)
results_2 <- train_test_neural_net(train_normalized_t2, test_normalized_t2, c(4), TRUE, NULL, test_original_pred_t2)
results_3 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(4), TRUE, NULL, test_original_pred_t3)
results_4 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(4), TRUE, NULL, test_original_pred_t4)
results_5 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(5), TRUE, NULL, test_original_pred_t4)
results_6 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(6), TRUE, NULL, test_original_pred_t4)
results_7 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(5), TRUE, NULL, test_original_pred_t3)
results_8 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(6), TRUE, NULL, test_original_pred_t3)
results_9 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(4,3), TRUE, NULL, test_original_pred_t4)
results_10 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(5,4), TRUE, NULL, test_original_pred_t4)
results_11 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(6,5), TRUE, NULL, test_original_pred_t4)
results_12 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(4,3), TRUE, NULL, test_original_pred_t3)
results_13 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(5,4), TRUE, NULL, test_original_pred_t3)
results_14 <- train_test_neural_net(train_normalized_t3, test_normalized_t3, c(6,5), TRUE, NULL, test_original_pred_t3)
results_15 <- train_test_neural_net(train_normalized_t4, test_normalized_t4, c(4), FALSE, "tanh", test_original_pred_t4)


# Combine metrics into a single data frame
all_results <- rbind(results_1, results_2, results_3, results_4, results_5, results_6, results_7, results_8, results_9, results_10, results_11, results_12, results_13, results_14, results_15)

# Display the table
View(all_results)

# Extracting the statistical indices for the best MLP network (results_15)
rmse_best <- results_15$RMSE
mae_best <- results_15$MAE
mape_best <- results_15$MAPE
smape_best <- results_15$sMAPE

# Printing out the statistical indices
cat("RMSE:", rmse_best, "\n")
cat("MAE:", mae_best, "\n")
cat("MAPE:", mape_best, "%\n")
cat("sMAPE:", smape_best, "%\n")

