## ------------------------------------------------------------------------------------------------------------------------
# set up the libraries
library(tidyverse)
library(readxl) 
library(here) # sets path to your folder directory
library(leaps) # for variable selection methods
library(ggplot2)
library(caret)
library(nnet) # for multinomial logistic regression
library(pROC) # for ROC curve
library(car) # for stress-testing regression models
library(rpart) # for regression trees
library(rpart.plot) # for regression tree plot
library(vip) # for feature importance
library(randomForest)
library(janitor) # to clean messy column names
library(knitr) # to convert from qmd to r script


## ------------------------------------------------------------------------------------------------------------------------
# load the data
pj_data <- read_xlsx(here("data", "project_data.xlsx")) 
# the data file is in a folder named 'data'
head(pj_data)
str(pj_data)
summary(pj_data)


## ------------------------------------------------------------------------------------------------------------------------
# clean messy column names
pj_data <- clean_names(pj_data) 

# remove any missing values
pj_data <- na.omit(pj_data) 


## ------------------------------------------------------------------------------------------------------------------------
# Factorize the categorical variables
# 'target' is the only categorical variable
col <- c("target")
pj_data [, col] <- lapply(pj_data[, col], factor)
str(pj_data) # confirm categorization of the column
levels(pj_data$target)
table(pj_data$target)


## ------------------------------------------------------------------------------------------------------------------------
# Let's take a look at the distribution of our variable of interest
pj_data |>
  ggplot(aes(x =target, fill = target))+
           geom_bar()+
  theme_minimal()+
  labs( title = "Distribution of Student Outcomes",
        x = "Outcome",
        y = "Count")


## ------------------------------------------------------------------------------------------------------------------------
# set reference level
pj_data$target <- relevel(pj_data$target, ref = "Graduate" )

# confirm
levels(pj_data$target)


## ------------------------------------------------------------------------------------------------------------------------
set.seed(123)  # setting a seed for reproducibility of the random split
index <- createDataPartition(pj_data$`target`, p = 0.3, list = FALSE)
training_data <- pj_data[-index, ]
testing_data <- pj_data[index, ]


## ------------------------------------------------------------------------------------------------------------------------
summary(training_data$`target`)
summary(testing_data$`target`)


## ------------------------------------------------------------------------------------------------------------------------
cat("Training rows:", nrow(training_data), "\n")
cat("Testing rows:", nrow(testing_data), "\n")


## ------------------------------------------------------------------------------------------------------------------------
prop.table(table(training_data$target))
prop.table(table(testing_data$target))


## ------------------------------------------------------------------------------------------------------------------------
# We first fit a temporary linear model just to extract VIF values - this 
# is a common workaround since vif() requires a linear model object
temp_model <- lm(as.numeric(target) ~ ., data = training_data)
vif_values <- vif(temp_model) # get the vif values from the model
vif_values[vif_values > 5] # show variables that have vif values greater 
# than 5 (potential collinearity issue)


## ------------------------------------------------------------------------------------------------------------------------
model_0 <- multinom(target~., data = training_data, maxit = 500)
# maxit = 500 gives the optimizer more iterations to converge
summary(model_0)


## ------------------------------------------------------------------------------------------------------------------------
# model fit: 
AIC(model_0) # lower AIC is better

# McFadden's Pseudo R-squared
# Formula: 1 - (log-likelihood of full model / log-likelihood of null model)
null_model <- multinom(target ~ 1, data = training_data, maxit = 500)
mcfadden_r2 <- 1 - (logLik(model_0) / logLik(null_model))
cat("McFadden's Pseudo R²:", round(as.numeric(mcfadden_r2), 4), "\n") # print
# the pseudo R squared value


## ------------------------------------------------------------------------------------------------------------------------
# make predictions on the test data
predicted_class <- predict(model_0, newdata = testing_data) 

# confusion matrix
conf_matrix <- confusionMatrix(predicted_class, testing_data$target)
conf_matrix


## ------------------------------------------------------------------------------------------------------------------------
# Relative Risk Ratios: exp(coefficients)
rrr <- exp(coef(model_0))
rrr


## ------------------------------------------------------------------------------------------------------------------------
importance <- varImp(model_0)
# rank the variables by importance
importance_df <- importance |>
  rownames_to_column(("Variable")) |>
  arrange(desc(Overall))

# view the variables df
head(importance_df)

# plot the variables
importance_df |>
  slice_head(n = 15) |>
  ggplot(aes(x = reorder(Variable,Overall), y = Overall))+
  geom_col(fill ="steelblue") +
  coord_flip()+
  theme_minimal()+
  labs(title = "Top 15 Most Important Predictors of Student Outcome",
       x = "Variable",
       y = "Importance Score")



## ------------------------------------------------------------------------------------------------------------------------
# classification tree model
model1 <- rpart(formula = target ~ .-target, 
                data= training_data, 
                method= "class")
# plot the model
rpart.plot(model1)
head(training_data)


## ------------------------------------------------------------------------------------------------------------------------
# make predictions on the test data
predicted_class_1 <- predict(model1, newdata = testing_data) 

testing_data$predicted_class_1 <- predict(model1, newdata = testing_data, 
                                          type = "class")
head(testing_data)

# confusion matrix
conf_matrix_1 <- confusionMatrix(testing_data$predicted_class_1, 
                                 testing_data$target)
conf_matrix_1



## ------------------------------------------------------------------------------------------------------------------------
vip(model1)


## ------------------------------------------------------------------------------------------------------------------------
model2 <-rpart(formula = target ~ ., data = training_data, method = "anova")
# we use 'anova' when developing a regression tree
rpart.plot(model2)


## ------------------------------------------------------------------------------------------------------------------------
# make predictions on the test data
predicted_class_2 <- predict(model2, newdata = testing_data)

# Apply the model to testing data
testing_data$predicted_class_2 <- predict(model2, newdata = testing_data, 
                                          type = "vector")
head(data.frame(testing_data$target, testing_data$predicted_class_2))

# Convert both to numeric before calculating
target_num <- as.numeric(as.character(testing_data$target))
pred_num <- as.numeric(as.character(testing_data$predicted_class_2))

# Calculate RSS
rss <- sum((target_num - pred_num)^2)
rss


## ------------------------------------------------------------------------------------------------------------------------
vip(model2)


## ------------------------------------------------------------------------------------------------------------------------
plotcp(model2)


## ------------------------------------------------------------------------------------------------------------------------
pruned1 <- rpart(formula = target ~ .,data= training_data, method  = "anova", 
                 control = list(cp = 0.022, xval = 10)) 
                  # xval = 10 is a common validation number
rpart.plot(pruned1)


## ------------------------------------------------------------------------------------------------------------------------
pruned2 <- rpart(formula = target ~ .,data= training_data, 
                 method  = "anova",  
                 control = list(cp = 0.014, 
                                xval = 10)) # xval = 10 is a common validation number
rpart.plot(pruned2)


## ------------------------------------------------------------------------------------------------------------------------
rf_model1 <- randomForest(target ~ ., data=training_data, 
                          mtry = 8, importance=TRUE)

# The default mtry for classification is the square root of the 
# number of predictors. We have 36 predictors, therefore 9 would 
# be a good value for mtry.

rf_model1 


## ------------------------------------------------------------------------------------------------------------------------
table(training_data$target)


## ------------------------------------------------------------------------------------------------------------------------
# Let's set the sample size based on our smallest class (Enrolled = 555)
# This forces the model to treat all classes with equal weight
rf_model1_balanced <- randomForest(
  target ~ ., 
  data = training_data, 
  mtry = 8, 
  importance = TRUE,
  sampsize = c("Graduate" = 555, "Dropout" = 555, "Enrolled" = 555)
)

print(rf_model1_balanced)


## ------------------------------------------------------------------------------------------------------------------------
# predict using the balanced random forest model
predicted_class_3 <- predict(rf_model1_balanced, newdata = testing_data)

# confusion matrix
brf_cm <- confusionMatrix(predicted_class_3, testing_data$target)
brf_cm


## ------------------------------------------------------------------------------------------------------------------------
varImpPlot(rf_model1_balanced)
importance(rf_model1_balanced)

# variable importance plot
vip(rf_model1_balanced)


## ------------------------------------------------------------------------------------------------------------------------
# Let's create a data frame with the Sensitivity scores we've gathered
comparison_data <- data.frame(
  Model = c("Multinomial Logistic", "Classification Tree", 
            "Balanced Random Forest"),
  Graduate = c(0.937, 0.890, 0.847),
  Dropout = c(0.785, 0.759, 0.730),
  Enrolled = c(0.276, 0.372, 0.544) # RF scores from the balanced OOB
)

# Reshape for plotting
plot_data <- pivot_longer(comparison_data, 
                          cols = c("Graduate", "Dropout", "Enrolled"), 
                          names_to = "Class", 
                          values_to = "Sensitivity")

# Generate a comparison plot
ggplot(plot_data, aes(x = Class, y = Sensitivity, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Comparison: Sensitivity by Student Outcome",
       subtitle = "The Balanced Random Forest is the only model that 
       effectively 'catches' Enrolled students",
       y = "Sensitivity (Recall)",
       x = "Student Status") +
  scale_fill_brewer(palette = "Set1") +
  geom_text(aes(label = round(Sensitivity, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, 
            size = 3)



## ------------------------------------------------------------------------------------------------------------------------
purl("project_code.qmd")

