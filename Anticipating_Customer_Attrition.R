# Loading the necessary packages
library(gridExtra)
library(tidyverse)
library(ggplot2)
library(GGally)
library(MASS)
library(smotefamily)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(glmnet)
library(caret)  
library(gbm)    

# Load data
data <- read.csv("train.csv")

# Explore data
summary(data)
str(data)

# Feature engineering

# Feature for the Ratio of "daysInactive" to "activeSince"
data$inactive_ratio <- data$daysInactive / data$activeSince

# Combine "daysInactiveAvg" and "daysInactiveSD" into a Single Feature
data$inactive_variability <- data$daysInactiveAvg + data$daysInactiveSD

# Combine "productCategories" and "productViews" into a Single Feature for Average Diversity
data$avg_product_diversity <- (data$productViews + data$productCategories) / 2

# Create a New Feature by Categorizing "timeOfDay"
# Assuming "timeOfDay" is in POSIXct format
data$time_of_day <- cut(data$timeOfDay, breaks = c(-Inf, 6, 12, 18, Inf), labels = c("Night", "Morning", "Afternoon", "Evening"), include.lowest = TRUE)

# Combine "duration" with "visits" to Create a Feature for Total Time Spent
data$total_time_spent <- data$duration * data$visits

# Sum Clicks Across Different Product Categories to Create an Overall Engagement Score
data <- data %>%
  group_by(id) %>%
  mutate(overall_engagement_score = sum(clicks, na.rm = TRUE))

# Group Similar Product Categories to Reduce Dimensionality
category_mapping <- list(
  Fashion = c("clicksClothing", "clicksShoes"),
  Tech = c("clicksElectronics", "clicksWatches"),
  Literature = c("clicksBooks", "clicksMovies", "clicksMusic"),
  Home = c("clicksKitchen", "clicksHome", "clicksGarden", "clicksPet", "clicksFood"),
  Toys = c("clicksToys"),
  Tools = c("clicksTools", "clicksAutomotive", "clicksOutdoors", "clicksHandmade", "clicksSports", "clicksScience", "clicksIndustrial")
)

for (group in names(category_mapping)) {
  if (all(category_mapping[[group]] %in% colnames(data))) {
    data[[group]] <- rowSums(data[, category_mapping[[group]]])
  } else {
    print(paste("Columns for", group, "not found in data."))
  }
}

# Remove excessive features and features that were combined
columns_to_remove <- c(
  "clicksClothing", "clicksShoes", "clicksElectronics", "clicksWatches", 
  "clicksBooks", "clicksMovies", "clicksMusic", "clicksJewelry", 
  "clicksKitchen", "clicksHome", "clicksGarden", "clicksPet", 
  "clicksFood", "clicksHealth", "clicksToys", "clicksTools", 
  "clicksAutomotive", "clicksOutdoors", "clicksHandmade", 
  "clicksSports", "clicksScience", "clicksIndustrial"
)
data <- data[, !(names(data) %in% columns_to_remove)]

summary(data)

# Missing value imputation
mean_inactive_ratio <- mean(data$inactive_ratio, na.rm = TRUE)
data$inactive_ratio[is.na(data$inactive_ratio)] <- mean_inactive_ratio

################### MODEL ###################

# Split the data into training and testing sets

library(caret)
library(pROC)

set.seed(123)
train_index <- createDataPartition(data$churn, p = 0.6, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the GBM model
gbm_model <- gbm(churn ~ ., data = train_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 4)

# Make predictions on the testing set
predictions <- predict(gbm_model, newdata = test_data, n.trees = 100, type = "response")

# Convert predicted probabilities to binary predictions
binary_predictions <- ifelse(predictions > 0.5, 1, 0)

# Evaluate performance using confusion matrix
conf_matrix <- table(test_data$churn, binary_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

# Calculate precision
precision <- conf_matrix[2, 2] / sum(binary_predictions)
print(paste("Precision:", precision))

# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / sum(test_data$churn)
print(paste("Recall (Sensitivity):", recall))

# Calculate F1-score
f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 Score:", f1_score))


#################### Plotting the AUC graph ###################

library(pROC)

# Calculate ROC curve
roc_curve <- roc(test_data$churn, predictions)

# Calculate AUC value
auc_value <- auc(roc_curve)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# Add AUC value to the plot
text(0.8, 0.2, paste("AUC =", round(auc_value, 3)), col = "red", cex = 1.2)

# Add a legend
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "red", lty = 1, cex = 0.8)

################### Testing Unseen Data ###################

# Load the test data
test_data_new <- read.csv("test.csv")

# Feature engineering for test data

# Feature for the Ratio of "daysInactive" to "activeSince"
test_data_new$inactive_ratio <- test_data_new$daysInactive / test_data_new$activeSince

# Combine "daysInactiveAvg" and "daysInactiveSD" into a Single Feature
test_data_new$inactive_variability <- test_data_new$daysInactiveAvg + test_data_new$daysInactiveSD

# Combine "productCategories" and "productViews" into a Single Feature for Average Diversity
test_data_new$avg_product_diversity <- (test_data_new$productViews + test_data_new$productCategories) / 2

# Create a New Feature by Categorizing "timeOfDay"
# Assuming "timeOfDay" is in POSIXct format
test_data_new$time_of_day <- cut(test_data_new$timeOfDay, breaks = c(-Inf, 6, 12, 18, Inf), labels = c("Night", "Morning", "Afternoon", "Evening"), include.lowest = TRUE)

# Combine "duration" with "visits" to Create a Feature for Total Time Spent
test_data_new$total_time_spent <- test_data_new$duration * test_data_new$visits

# Sum Clicks Across Different Product Categories to Create an Overall Engagement Score
test_data_new <- test_data_new %>%
  group_by(id) %>%
  mutate(overall_engagement_score = sum(clicks, na.rm = TRUE))

# Group Similar Product Categories to Reduce Dimensionality
category_mapping <- list(
  Fashion = c("clicksClothing", "clicksShoes"),
  Tech = c("clicksElectronics", "clicksWatches"),
  Literature = c("clicksBooks", "clicksMovies", "clicksMusic"),
  Home = c("clicksKitchen", "clicksHome", "clicksGarden", "clicksPet", "clicksFood"),
  Toys = c("clicksToys"),
  Tools = c("clicksTools", "clicksAutomotive", "clicksOutdoors", "clicksHandmade", "clicksSports", "clicksScience", "clicksIndustrial")
)

for (group in names(category_mapping)) {
  if (all(category_mapping[[group]] %in% colnames(test_data_new))) {
    test_data_new[[group]] <- rowSums(test_data_new[, category_mapping[[group]]])
  } else {
    print(paste("Columns for", group, "not found in test data."))
  }
}

# Remove excessive features and features that were combined
columns_to_remove_test <- c(
  "clicksClothing", "clicksShoes", "clicksElectronics", "clicksWatches", 
  "clicksBooks", "clicksMovies", "clicksMusic", "clicksJewelry", 
  "clicksKitchen", "clicksHome", "clicksGarden", "clicksPet", 
  "clicksFood", "clicksHealth", "clicksToys", "clicksTools", 
  "clicksAutomotive", "clicksOutdoors", "clicksHandmade", 
  "clicksSports", "clicksScience", "clicksIndustrial"
)
test_data_new <- test_data_new[, !(names(test_data_new) %in% columns_to_remove_test)]

summary(test_data_new)

# Missing value imputation
mean_inactive_ratio_test <- mean(test_data_new$inactive_ratio, na.rm = TRUE)
test_data_new$inactive_ratio[is.na(test_data_new$inactive_ratio)] <- mean_inactive_ratio_test

# Make predictions on the test set
predictions_2 <- predict(gbm_model, newdata = test_data_new, n.trees = 100, type = "response")

# Create a submission data frame
submission <- data.frame(id = test_data_new$id, churn = predictions_2)

# Write the submission file to CSV
write.csv(submission, file = "submission.csv", row.names = FALSE)