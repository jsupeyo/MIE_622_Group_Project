## ------------------------------------------------------------------------------------------------------------
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


## ------------------------------------------------------------------------------------------------------------
# load the data
pj_data <- read_xlsx(here("data", "project_data.xlsx")) 
# the data file is in a folder named 'data'
head(pj_data)
str(pj_data)
summary(pj_data)


## ------------------------------------------------------------------------------------------------------------
# clean messy column names
pj_data <- clean_names(pj_data) 

# remove any missing values
pj_data <- na.omit(pj_data) 


## ------------------------------------------------------------------------------------------------------------
colnames(pj_data)


## ------------------------------------------------------------------------------------------------------------
# Factorize the categorical variables
# 'target' is the only categorical variable
col <- c("target")
pj_data [, col] <- lapply(pj_data[, col], factor)
str(pj_data) # confirm categorization of the column
levels(pj_data$target)
table(pj_data$target)


## ------------------------------------------------------------------------------------------------------------
# Let's take a look at the distribution of our variable of interest
pj_data |>
  ggplot(aes(x =target, fill = target))+
           geom_bar()+
  theme_minimal()+
  labs( title = "Distribution of Student Outcomes",
        x = "Outcome",
        y = "Count")


## ------------------------------------------------------------------------------------------------------------
# set reference level
pj_data$target <- relevel(pj_data$target, ref = "Graduate" )

# confirm
levels(pj_data$target)


## ------------------------------------------------------------------------------------------------------------
set.seed(123)  # setting a seed for reproducibility of the random split
index <- createDataPartition(pj_data$`target`, p = 0.3, list = FALSE)
training_data <- pj_data[-index, ]
testing_data <- pj_data[index, ]


## ------------------------------------------------------------------------------------------------------------
summary(training_data$`target`)
summary(testing_data$`target`)


## ------------------------------------------------------------------------------------------------------------
cat("Training rows:", nrow(training_data), "\n")
cat("Testing rows:", nrow(testing_data), "\n")


## ------------------------------------------------------------------------------------------------------------
prop.table(table(training_data$target))
prop.table(table(testing_data$target))


## ------------------------------------------------------------------------------------------------------------
# We first fit a temporary linear model just to extract VIF values - this 
# is a common workaround since vif() requires a linear model object
temp_model <- lm(as.numeric(target) ~ ., data = training_data)
vif_values <- vif(temp_model) # get the vif values from the model
vif_values[vif_values > 5] # show variables that have vif values greater 
# than 5 (potential collinearity issue)


## ------------------------------------------------------------------------------------------------------------
# Compute a correlation matrix for all numeric predictors
# (excluding Target)
numeric_vars <- training_data |> select(-target)
cor_matrix <- cor(numeric_vars)

# Visualize it — easier to spot pairs of correlated variables
library(corrplot)
corrplot(cor_matrix, method = "color", tl.cex = 0.6)


## ------------------------------------------------------------------------------------------------------------
# Parental background — keep mothers_qualification & fathers_occupation
# Previous qualification — keep previous_qualification_grade
# Curricular units — keep 2nd sem approved & grade only
# Economic indicators — keep unemployment_rate

vars_to_remove <- c(
  "fathers_qualification",
  "mothers_occupation",
  "previous_qualification",
  "curricular_units_1st_sem_credited",
  "curricular_units_1st_sem_enrolled",
  "curricular_units_1st_sem_evaluations",
  "curricular_units_1st_sem_approved",
  "curricular_units_1st_sem_grade",
  "curricular_units_1st_sem_without_evaluations",
  "curricular_units_2nd_sem_credited",
  "curricular_units_2nd_sem_enrolled",
  "curricular_units_2nd_sem_evaluations",
  "curricular_units_2nd_sem_without_evaluations",
  "inflation_rate",
  "gdp"
)

training_data <- training_data |> select(-all_of(vars_to_remove))
testing_data  <- testing_data  |> select(-all_of(vars_to_remove))


## ------------------------------------------------------------------------------------------------------------
temp_model_2 <- lm(as.numeric(target) ~ ., data = training_data)
vif(temp_model_2)
# All VIF values should now be below 5


## ------------------------------------------------------------------------------------------------------------
model_0 <- multinom(target~., data = training_data, maxit = 500)
# maxit = 500 gives the optimizer more iterations to converge
summary(model_0)


## ------------------------------------------------------------------------------------------------------------
# model fit: 
AIC(model_0) # lower AIC is better

# McFadden's Pseudo R-squared
# Formula: 1 - (log-likelihood of full model / log-likelihood of null model)
null_model <- multinom(target ~ 1, data = training_data, maxit = 500)
mcfadden_r2 <- 1 - (logLik(model_0) / logLik(null_model))
cat("McFadden's Pseudo R²:", round(as.numeric(mcfadden_r2), 4), "\n") # print
# the pseudo R squared value


## ------------------------------------------------------------------------------------------------------------
# Odds ratios a.k.a Relative Risk Ratios: exp(coefficients)
odds_ratios <- exp(coef(model_0))
odds_ratios


## ------------------------------------------------------------------------------------------------------------
# Create a clean odds ratio dataframe for plotting
or_df <- data.frame(
  Variable = colnames(coef(model_0)),
  OR_Dropout = as.numeric(exp(coef(model_0))["Dropout", ])
) |>
  filter(Variable != "(Intercept)") |>  # remove intercept
  mutate(
    Direction = ifelse(OR_Dropout > 1, "Risk Factor", "Protective Factor")
  ) |>
  arrange(desc(OR_Dropout))

# Plot
ggplot(or_df, aes(x = reorder(Variable, OR_Dropout), 
                   y = OR_Dropout, 
                   fill = Direction)) +
  geom_col() +
  geom_hline(yintercept = 1, 
             linetype = "dashed", 
             color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("Risk Factor" = "tomato", 
                                "Protective Factor" = "steelblue")) +
  theme_minimal() +
  labs(title = "Variable Importance Plot",
    subtitle = "Odds Ratios: Dropout vs. Graduate",
       x = "Variable",
       y = "Odds Ratio",
       fill = "")


## ------------------------------------------------------------------------------------------------------------
# make predictions on the test data
predicted_class <- predict(model_0, newdata = testing_data) 

# confusion matrix
conf_matrix <- confusionMatrix(predicted_class, testing_data$target)
conf_matrix


## ------------------------------------------------------------------------------------------------------------
# classification tree model
model1 <- rpart(formula = target ~ .-target, 
                data= training_data, 
                method= "class")
# plot the model
rpart.plot(model1)
head(training_data)


## ------------------------------------------------------------------------------------------------------------
# make predictions on the test data
predicted_class_1 <- predict(model1, newdata = testing_data) 

testing_data$predicted_class_1 <- predict(model1, newdata = testing_data, 
                                          type = "class")
head(testing_data)

# confusion matrix
conf_matrix_1 <- confusionMatrix(testing_data$predicted_class_1, 
                                 testing_data$target)
conf_matrix_1



## ------------------------------------------------------------------------------------------------------------
vip(model1)+
  geom_col(fill="steelblue")+
  labs( title = "Variable Importance Plot",
        subtitle = "Classification Model")+
  theme_minimal()


## ------------------------------------------------------------------------------------------------------------
set.seed(123) # for reproducibility
rf_model1 <- randomForest(target ~ ., 
                          data=training_data, 
                          mtry = 5,
                          importance=TRUE)

# The default mtry for classification is the square root of the 
# number of predictors. We have 21 predictors, therefore 5 would 
# be a good value for mtry.

rf_model1 


## ------------------------------------------------------------------------------------------------------------
table(training_data$target)


## ------------------------------------------------------------------------------------------------------------
# Let's set the sample size based on our smallest class (Enrolled = 555)
# This forces the model to treat all classes with equal weight
rf_model1_balanced <- randomForest(
  target ~ ., 
  data = training_data, 
  mtry = 5, 
  importance = TRUE,
  sampsize = c("Graduate" = 555, "Dropout" = 555, "Enrolled" = 555)
)

print(rf_model1_balanced)


## ------------------------------------------------------------------------------------------------------------
# predict using the balanced random forest model
predicted_class_3 <- predict(rf_model1_balanced, newdata = testing_data)

# confusion matrix
brf_cm <- confusionMatrix(predicted_class_3, testing_data$target)
brf_cm


## ------------------------------------------------------------------------------------------------------------
varImpPlot(rf_model1_balanced)
importance(rf_model1_balanced)

# variable importance plot
vip(rf_model1_balanced)+
  geom_col(fill = "steelblue")+
  labs( title = "Variable Importance Plot",
        subtitle = "Balanced Random Forest Model")+
  theme_minimal()


## ------------------------------------------------------------------------------------------------------------
# Let's create a data frame with the Sensitivity scores we've gathered
comparison_data <- data.frame(
  Model = c("Multinomial Logistic", "Classification Tree", 
            "Balanced Random Forest"),
  Graduate = c(0.925, 0.890, 0.863),
  Dropout = c(0.764, 0.759, 0.752),
  Enrolled = c(0.209, 0.372, 0.536) # RF scores from the balanced OOB
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



## ------------------------------------------------------------------------------------------------------------
purl("project_code.qmd")

