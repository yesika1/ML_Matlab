
# ---------------------------------------------------------------------------------------- #
#
# Exercise DataCamp
#
# ---------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------- #
#
# Credit Risk Modeling in R
# 
# Loan Default:
# -----------
#   A bank lends money to a borrower, individual or company, the bank transfer the entire amount of the loan to the borrower. 
#   The borrower reimberse this amount in smaller chunks, including interest payments and fees over time (montly, quaterly or yearly).
#   There is a certain Risk that the borrower will not be able to fully reimberse the loan. This result in a loss for the bank. 
# 
# Components of Expected loss (EL):
# -------------
#  a. Probability of Default (PD): Probability that the borrower with fail to make a fully repayment.
# 
#  b. Exposure At Default (EAD): Expected value of the Loan at the time of default. It is the amount of the loan that still needs to be repay at the time of default.
# 
#  c. Loss Given Default (LGD): Amount of the Loss if the reset the defaut. 
# 
#         EL: PD * EAD * LGD
# 
# Type of Information used by banks:
# ----------
#   
#  a. Aplication Information: Income, marital status
#  b. Behavioral Information: Pass behavior of customers. Ex: current account balance, payment arrears in account history
# 
# Loan status: 1: if customer defaulted, 0: customer not defaulted.
# 
# ---------------------------------------------------------------------------------------- #


#============================================================
# Libraries
#============================================================
library(gmodels) # used for crossTable- categorical variables
library(ggplot2) # used for graphs



#============================================================
# Importing data and first overview
#============================================================
str(df)
# loan_status: where 1 represents a default, and 0 represents non-default.
head(df) # show first 6 observations
summary(df) # tell me missing values per var

# For categorical data:
#======
CrossTable(df$catVar1)
# Show number of cases and proportions

CrossTable(x = df$catVar1, y= df$target, prop.r=TRUE, prop.c = F, prop.t = F, prop.chisq = F )
# Show relationship of var and response (row-wise proportions)

pl1 <- ggplot(df, aes(target)) + geom_bar( alpha=0.5,aes(fill=target)) +ggtitle("Figure 4. Barplot Target Feature (var0)"); pl1
# Barplot Target Feature

bar1 <-ggplot(df, aes(x=var1, color=target)) + geom_bar(aes(fill=target), position = 'fill') +ggtitle("Figure 7. Barplot var1 by target factors") ; bar1
# Barplot. proportion do not change considerable by level of target feature, will not be transformed in factor


# For numerical data:
#======



# histograms
p<-ggplot(df, aes(x=weight)) +  geom_histogram(color="black", fill="white")

pl3 <-ggplot(df, aes(y= target,x= var4)) + geom_histogram( aes( group=target, fill=factor(target), alpha=0.4, ), position="top") +ggtitle("Figure 5. Boxplot var4 by Target Feature (target)");pl3 

# boxplot var4 and target : no distinction
pl3 <-ggplot(df, aes(target,var4)) + geom_boxplot( aes( group=target, fill=factor(target), alpha=0.4)) +ggtitle("Figure 5. Boxplot var4 by Target Feature (target)");pl3 


# outliers
#=======
# After an histogram looks funny. Outliers are easy to observe in a scatter plot. 
ggplot(mapping=aes(x=seq_along(df$var), y=df$var.Length)) +
  geom_point()+ geom_text(label=rownames(df)) + labs(x = "Index")

# Expert judgement: Annual salaries > 3 million are outliers
index_outlier_expert <- which( df$var > 3000000) # getting index 
loan_data_expert <- df[-index_outlier_expert ,]  # new df removing rows with outliers index

# Rule of thumb (ROT): outlier if bigger than Q3+1.5* IQR
outlier_cutoff <- quantile(df$var, 0.75) + 1.5 * IQR(df$var)
index_outlier_ROT <- which(df$var > outlier_cutoff)
loan_data_ROT <- df[-index_outlier_ROT,] # new df removing outliers index

# histogram for Expert judgement and ROT
newdf$var


## Bivariate plot: to identify Bivariate  Outliers in two dimensions of the data.
# If outliers are observed for several variables. It's possible the outliers belong to the same observation.
# If so, there is even more reason to delete the observation because it is more likely that some information stored in it is wrong.

 

## Missing Values
#------
#NA: Not Available
# WE CAN: delete row(for few NA cases)/column, replace or Keep them.

# Delete Rows:
index_NA: which(is.na(df$var))
df_no_NA <- df[-c(index_NA), ]

# Delete column:
df_delete_var <- df
df_delete_var$var <- NULL

# Replace: median/mode imputation
index_NA: which(is.na(df$var))
df_replace <- df
df_replace$var[index_NA] <- median(df$var,na.rm=T)

mode <- function (x, na.rm)  { 
       xtab <- table(x) 
       xmode <- names(which(xtab == max(xtab))) 
      if (length(xmode) > 1) 
             xmode <- ">1 mode" 
          return(xmode) 
} 

df_replace$var[index_NA] <- mode(df$var,na.rm=T)

library(mice)
md.pattern(data) # tell: complete(all var=1) or missing by variable: it a var has 0, then there are missing values which is the first column

tempData <- mice(data,maxit=50,meth='pmm',seed=500) ## perform mice imputation, based on mean. (method="rf":random forests.)
summary(tempData)
tempData$imp$var # checking the imputed data for a var
completedData <- complete(tempData,1)

#m=5 refers to the number of imputed datasets. Five is the default value.
# meth='pmm' refers to the imputation method. In this case we are using predictive mean matching as imputation method. Other imputation methods can be used, type methods(mice)
#  mice package to do a quick and dirty imputation. 

miceMod <- mice(df, method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)

#Keep NA
#coarse classification: put variables in BINS (categories), includying NA category, for categoriacal just create a missing category. 
#Try and error from diff. bin ranges, using qunatile func. to identify breaks.

df$var_bins <- rep(NA, length(df$var)) # replicate func. to create a new var of NA with lenght of original var

df$var_bins[which(df$var <= 15)] <- "0-15" # splitting data of original var in bins
df$var_bins[which(df$var > 15 & df$var <= 30)] <- "15-30"
df$var_bins[which(df$var > 30 & df$var <= 45)] <- "30-45"
df$var_bins[which(df$var > 45)] <- "45+"
df$var_bins[which(is.na(df$var))] <- "Missing"
df$var_bins <- as.factor(df$var_bins) # converting new var in factor
plot(df$var_bins) # check if the bin are right or chnage the interval values



# Splitting data set
# Set seed of 567
set.seed(567)

# Store row numbers for training set: index_train
index_train <- sample(1:nrow(loan_data),2 / 3 * nrow(loan_data))

# Create training set: training_set
training_set <- loan_data[index_train, ]

# Create test set: test_set
test_set <- loan_data[-index_train,]

# Cross-validation
#evaluating: confusion matrix
#actual loan status vs. model prediction: No default (0), Default(1)
# Accuracy:% correctly classify instances =(TP+TN)/totalPopulation =(TP+TN)/(TP+FP+TN+FN)
# Sensitivity: % of BAD customers that are classify correctly = TPDefault/totalActualDefault= TP/(TP+FN)
# Specificity: % of GOOD customers that are classify correctly = TN-NonDefault/totalActualNonDefault = TN/(TN+FP)


#LOGISTIC REGRESSION MODEL

# Build a glm model with variable ir_cat as a predictor
# glm(y ~ x1 + x2, family = "binomial", data = data)
log_model_cat <- glm(loan_status ~ ir_cat+ var2,family = "binomial",data = training_set)

# Print the parameter estimates 
log_model_cat
log_model_cat$coefficients

#exp(0.5414484) 
# Compared to the reference category with interest rates between 0% and 8 (not in the equation), 
# the odds in favor of default/target change by a multiple of...exp of a coefficient

# Obtain significance levels using summary()
summary(log_model_multi)

#When you do include several variables and ask for the interpretation when a certain variable changes, it is assumed that the other variables remain constant, or unchanged. 
# You already looked at the parameter values, but that is not the only thing of importance. 
# Also important is the statistical significance of a certain parameter estimate. 
# The significance of a parameter is often refered to as a p-value, however in a model output you will see it denoted as Pr(>|t|)
# In glm, mild significance is denoted by a "." to very strong significance denoted by "***".
# When a parameter is not significant, this means you cannot assure that this parameter is significantly different from 0. 
# Statistical significance is important. In general, it only makes sense to interpret the effect on default for significant parameters.


# predict probability
test_case <- as_tibble(test_set[1,]); test_case #one record

predict(log_model_cat, newdata =test_case) #output
predictions<- predict(log_model_cat, newdata =test_set, type= 'response') #output prob.

range(predictions)
# After having obtained all the predictions for the test set elements, 
# it is useful to get an initial idea of how good the model is at discriminating 
# by looking at the range of predicted probabilities. 
# A small range means that predictions for the test set cases do not lie far apart, 
# and therefore the model might not be very good at discriminating good from bad customers. 
# With low default percentages, you will notice that in general, very low probabilities of default are predicted. 

# Make a binary predictions-vector using a cut-off of 15%
cutoff <- 0.15
class_pred_logit <- ifelse(predictions_logit > cutoff, 1, 0) # testing whether a certain value in the predictions-vector is bigger than 0.15. If this is TRUE, R returns "1" (specified in the second argument), else 0. 

# Construct a confusion matrix: Specify true values first, then your predicted values.
tab_class_logit <- table(test_set$loan_status, pred_cutoff_15)

# Compute the classification accuracy for all three models
acc_logit <- sum(diag(tab_class_logit)) / nrow(test_set)

# Comparing two cut-offs
# Moving from a cut-off of 15% to 20%...Accuracy increases, sensitivity decreases and specificity increases.




#### DECISION TREES
#=========================

# The Gini-measure of the root node is given below
#Gini node = 2 * proportion of defaults in this node * proportion of non-defaults in this node. 
gini_root <- 2 * 89 / 500 * 411 / 500

# Compute the Gini measure for the left leaf node
gini_ll <- 2 * 401/446 *45/446

# Compute the Gini measure for the right leaf node
gini_rl <- 2 * 10/54 * 44/54

# Compute the gain
# Gain = gini_root - (prop(cases left leaf) * gini_left) - (prop(cases right leaf * gini_right))

gain <- gini_root - 446 / 500 * gini_ll - 54 / 500 * gini_rl

# compare the gain-column in small_tree$splits with our computed gain, multiplied by 500, and assure they are the same
small_tree$splits #Information regarding the split
#improve= 49.10042 # improve is an alternative metric for gain, simply obtained by multiplying gain by the number of cases in the data set.
improve <- gain * 500; improve

# Load package rpart in your workspace.
library(rpart)

# Change the code provided in the video such that a decision tree is constructed using the undersampled training set. Include rpart.control to relax the complexity parameter to 0.001.
tree_undersample <- rpart(loan_status ~ ., method = "class",
                          data =  undersampled_training_set,
                          control = rpart.control(cp = 0.001))

#cp, which is the complexity parameter, is the threshold value for a decrease in overall lack of fit for any split. If cp is not met, further splits will no longer be pursued. cp's default value is 0.01, but for complex problems, it is advised to relax cp.

# Plot the decision tree
plot(tree_undersample,uniform = TRUE) # uniform = TRUE to get equal-sized branches
text(tree_undersample) # Add labels to the decision tree


## Changing the prior probabilities:
#=====
# Change the code below such that a tree is constructed with adjusted prior probabilities.
# argument in rpart to include prior probabities: parms = list(prior=c(non_default_proportion, default_proportion))
non_default_proportion = 0.7
default_proportion=0.3
tree_prior <- rpart(loan_status ~ ., method = "class",
                    data = training_set,
                    control = rpart.control(cp = 0.001),
                    parms = list(prior=c(non_default_proportion, default_proportion)))

## Including a loss matrix:
#======
# Change the code below such that a decision tree is constructed using a loss matrix penalizing 10 times more heavily for misclassified defaults.
# You want to stress that misclassifying a default as a non-default should be penalized more heavily. 
# modify argument parms: parms = list(loss = matrix(c(0, cost_def_as_nondef, cost_nondef_as_def, 0), ncol=2))
# Doing this, you are constructing a 2x2-matrix with zeroes on the diagonal and changed loss penalties off-diagonal. The default loss matrix is all ones off-diagonal.
#parms = list(loss = matrix(c(0, 10, 1, 0), ncol = 2))

cost_def_as_nondef =10
cost_nondef_as_def =1
tree_loss_matrix <- rpart(loan_status ~ ., method = "class",
                          data =  training_set,
                          control = rpart.control(cp = 0.001),
                          parms = list(loss = matrix(c(0, cost_def_as_nondef, cost_nondef_as_def, 0), ncol=2)))


# Plot the decision tree
plot(tree_loss_matrix,uniform = TRUE) # uniform = TRUE to get equal-sized branches
text(tree_loss_matrix) # Add labels to the decision tree


### Pruning the tree
#=====

# tree_prior is loaded in your workspace
tree_prior

# Use printcp() to identify for which complexity parameter the cross-validated error rate is minimized.
printcp(tree_prior) 
tree_prior$cptable
tree_prior$cptable[,c('CP','xerror')]

# Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_prior)

#Use which.min() to identify which row in tree_prior$cptable has the minimum cross-validated error "xerror". Assign this to index.
# Create an index for of the row with the minimum xerror
index <- which.min(tree_prior$cptable[ , "xerror"])

# Create tree_min
tree_min <- tree_prior$cptable[index, "CP"]

#  Prune the tree using tree_min
ptree_prior <- prune(tree_prior, cp = tree_min)

# Use prp() to plot the pruned tree
library(rpart.plot)
prp(ptree_prior)



### Pruning the tree with the loss matrix when Min CP does not prune 
#===========

# Prune the tree that was built using a loss matrix in order to penalize 
# misclassified defaults more than misclassified non-defaults.

# set a seed and run the code to construct the tree with the loss matrix again
set.seed(345)
tree_loss_matrix  <- rpart(loan_status ~ ., method = "class", data = training_set,
                           parms = list(loss=matrix(c(0, 10, 1, 0), ncol = 2)),
                           control = rpart.control(cp = 0.001))

# Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_loss_matrix)
# Looking at the cp-plot, you will notice that pruning the tree using the minimum cross-validated error will lead to a tree that is as big as the unpruned tree, as the cross-validated error reaches its minimum for cp = 0.001

# Prune the tree using cp = 0.0012788
# Because you would like to make the tree somewhat smaller. For this complexity parameter, the cross-validated error approaches the minimum observed error.
ptree_loss_matrix <- prune(tree_loss_matrix, cp = 0.0012788)

# Use prp() and argument extra = 1 to plot the pruned tree
prp(ptree_loss_matrix, extra = 1)
