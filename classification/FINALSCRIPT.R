library(readr)
library(caret)
library(dplyr)
library(tibble)
library(purrr)
library(corrplot)
library(DescTools)
client_attrition_train <- read_csv("client_attrition_train.csv")
View(client_attrition_train)
head(client_attrition_train)
nrow(client_attrition_train)
dim(client_attrition_train)
### we have NA values and Unknown values
#### we replace all NA value in Unknown to have one category then we decide
#to eliminate certain variable or replace the NA value with the mode or mean
### or create like a dummy so another level of that variable
#### replace NA, for the numeric mean and categorical with mode 
### how many NA in our dataset
library(dplyr)
colSums(is.na(client_attrition_train)) %>% 
  sort()
# customer sex = 1018
# total transaction amount 407
# customer age = 624
# customer salary range = 681

#mean in group closed   mean in group open 
#46.62196             46.25996 
## how many Unknown Values? 
unknown_count <- sapply(client_attrition_train, function(x) sum(x == "Unknown" | is.na(x)))
unknown_count <- sapply(client_attrition_train, function(x) sum(x == "Unknown"))
# customer education = 1519
# customer civil status = 749

#####  we eliminate the variable of customer_sex because to check correlation 
#between sex and account status 
install.packages("vcd")
library(vcd)
newsexcustomer <- na.omit(client_attrition_train$customer_sex)
d <- subset(client_attrition_train, client_attrition_train$customer_sex!="NA")


contingencytable <- table(client_attrition_train$customer_sex, client_attrition_train$account_status)
corr <- assocstats(contingencytable)$cramer
corr
#### the correlation is very small, this justifies us to eliminate this variable

###### how many "unknown" values
unknown_count <- sapply(client_attrition_train, function(x) sum(x == "Unknown"))


h <- sum(client_attrition_train$customer_salary_range == "NA")
h
na_count <- sum(is.na(client_attrition_train[, client_attrition_train$customer_salary_range]))




client_attrition_train[is.na(client_attrition_train)] <- "Unknown"
char_cols <- sapply(client_attrition_train, is.character)

# Replace NA values with "Unknown" in character columns
client_attrition_train[, char_cols][is.na(client_attrition_train[, char_cols])] <- "Unknown"
colSums(is.na(client_attrition_train)) %>% 
  sort()

#### we replace the NA value in total transaction amount and customer age with the mean 

# Replace NA values with the mean total transaction
newtotalamount <- na.omit(client_attrition_train$total_transaction_amount)
## correlation with attrition?
d <- subset(client_attrition_train, client_attrition_train$total_transaction_amount!="NA")
result <- t.test(client_attrition_train$total_transaction_amount ~ client_attrition_train$account_status)
#there is a significant difference between the two groups so we cannot eliminate the variable for the classification
# so we replace NA value with mean
round(mean(newtotalamount))
column_name <- "total_transaction_amount" 
client_attrition_train[is.na(client_attrition_train[, column_name]), column_name] <- round(mean(newtotalamount))
View(client_attrition_train)

#customer age  correlation with attrition?
newage <- na.omit(client_attrition_train$customer_age)
round(mean(newage))
d1 <- subset(client_attrition_train, client_attrition_train$customer_age!="NA")
result1 <- t.test(client_attrition_train$customer_age ~ client_attrition_train$account_status)
# we can't reject the null-hypothesis of no difference we can eliminate these variable, but
# intuitevily we want to mantain this variable for predictions
# so I replace the mean age in the missing values
column_name1 <- "customer_age"
client_attrition_train[is.na(client_attrition_train[, column_name1]), column_name1] <- round(mean(newage))





## so we can eliminate the variable sex we saw that there is no correlation with the client  attrition 
## on the other hand we have correlation with the total trnsaction amount so we should replace the NA value with the mean
round(mean(newtotalamount))
column_name <- "total_transaction_amount" 
client_attrition_train[is.na(client_attrition_train[, column_name]), column_name] <- round(mean(newtotalamount))

newdata <- client_attrition_train[,-c(2,3)]
dim(newdata)


##### we transform the NA value of customer salary range into Unknown 
colSums(is.na(newdata)) %>% 
  sort()

newdata[is.na(newdata)] <- "Unknown"

### so now we have newdata with no NA value at all but we have the only problem of Unknown
### we can decide if consider Unknown as a level of the variable 
### actually in the variable of salary range we believe that the people with lower or higher salary are 
### unlikely to say own salary 

unknown_count <- sapply(newdata, function(x) sum(x == "Unknown"))
unknown_count
# customer education = 1519  ORDINAL VARIABLE
# customer civil status = 749 CATEGORICAL VARIABLE
# customer salary range = 1711 ORDINAL VARIABLE

##### we eliminated the variable sex and replace the missing values of numerical
# variable with the mean

newdata

############################# PREPARATION of the single variable ###########à

str(newdata)
#Storing qualitative variables as factors
#qualitative variables may be nominal or ordinal.

##### nominal variable ###
# customer_civil_status
# account_status

newdata$customer_civil_status <- as.factor(newdata$customer_civil_status)

newdata$account_status <- as.factor(newdata$account_status)

levels(newdata$customer_civil_status)
levels(newdata$account_status)

######### ordinal variable ###
# customer_education
# customer_range_salary
## credit_card_classification


newdata$customer_education <- as.factor(newdata$customer_education)
levels(newdata$customer_education)

### customer_education
### Let's pretend the unknown level is the lowest
table(newdata$customer_education)
newdata$customer_education <- factor(newdata$customer_education,
                                     levels = c("Unknown",
                                                "Uneducated",
                                                "High School" ,
                                                "College",
                                                "Graduate",
                                                "Post-Graduate",
                                                "Doctorate"),
                                     ordered = TRUE)
levels(newdata$customer_education)

### credit_card_classification
newdata$credit_card_classification <- factor(newdata$credit_card_classification,
                                             levels = c("Blue",
                                                        "Silver",
                                                        "Gold",
                                                        "Platinum"),
                                             ordered = TRUE)
levels(newdata$credit_card_classification)
table(newdata$credit_card_classification)

#### customer_range_salary ####
table(newdata$customer_salary_range)
newdata$customer_salary_range <- factor(newdata$customer_salary_range,
                                        levels = c("Unknown",
                                                   "below 40K",
                                                   "40-60K",
                                                   "60-80K",
                                                   "80-120K",
                                                   "120K and more"),
                                        ordered = TRUE)
levels(newdata$customer_salary_range)
table(newdata$customer_salary_range)

set.seed(987654321)

data_training <- createDataPartition(newdata$account_status,
                                     p = 0.7, 
                                     list = FALSE) 

data_train <- newdata[c(data_training),]
data_test <- newdata[-c(data_training),]



summary_binary_class <- function(predicted_classes,
                                 real,
                                 level_positive = "open",
                                 level_negative = "closed") {
  # save the classification table
  ctable <- confusionMatrix(as.factor(predicted_classes), 
                            real, 
                            level_positive) 
  # extract selected statistics into a vector
  stats <- round(c(ctable$overall[1],
                   ctable$byClass[c(1:4, 7, 11)]),
                 5)
  # and return them as a function result
  return(stats)
}


data_train1 <- data_train[,-c(4,7,13)]
data_test1 <- data_test[,-c(4,7,13)]

library(readr)
realattrition_test <- read_csv("client_attrition_test.csv")
head(realattrition_test)



realattrition_testdefinitive <- realattrition_test[,-c(2,3,6,9,15)]

colSums(is.na(realattrition_testdefinitive)) %>% 
  sort()

newtotalamount1 <- na.omit(realattrition_testdefinitive$total_transaction_amount)

mean(newtotalamount1) # pretty close so we replace with the mean of train sample


round(mean(newtotalamount))
column_name <- "total_transaction_amount" 
realattrition_testdefinitive[is.na(realattrition_testdefinitive[, column_name]), column_name] <- round(mean(newtotalamount))
##### TOTAL TRANSACTION AMOUNT REPLACE WITH MEAN 

### how about customer salary range
realattrition_testdefinitive[is.na(realattrition_testdefinitive)] <- "Unknown"


ctrl_nocv <- trainControl(method = "none")
attrition_train_knn5 <- 
  train(account_status ~ ., 
        data = data_train1 %>% 
          # we exclude customerID
          dplyr::select(-customer_id),
        # model type - now knn!!
        method = "knn",
        # train control
        trControl = ctrl_nocv)

attrition_train_knn5
attrition_train_knn5$finalModel
attrition_train_knn5$finalModel$k # number 5 folds is the default number

trainforecasts <- predict(attrition_train_knn5,
                                           data_test1)
summary_binary_class(predicted_classes = trainforecasts,
                     real = data_test1$account_status)

attrition_train_knn5


data_realtest_knn5_forecasts <- predict(attrition_train_knn5,
                                      realattrition_testdefinitive )

dim(realattrition_testdefinitive)

ncol(data_realtest_knn5_forecasts)

data_realtest_knn5_forecasts <- as.data.frame(data_realtest_knn5_forecasts)

dim(Mdata_train_knn5_forecasts)

file_path <- "FINALPREDICTIONS.csv"

# Save the dataset as a CSV file
write.csv(Mdata_train_knn5_forecasts, file = file_path, row.names = FALSE)
