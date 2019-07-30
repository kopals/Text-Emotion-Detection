#pdf("c:/Users/Mohammad Opal/Downloads/Intelligent System/sentiment text analysis.pdf")
##reading first 1000 reviews
#reviews_text <- readLines("C:/Users/Mohammad opal/Downloads/ECE579_Project/text_emotion.csv",n = 1000 )
# converting the reviews_text character vector to a dataframe
#reviews_text<-data.frame(reviews_text)  
#x=reviews_text_sep=gsub('\"'," "tweet_id","sentiment","author","content"")  
#reviews_text_sep<-separate(data = reviews_text, col='"tweet_id","sentiment","author","content"', into=c("tweet_id","sentiment","author","content"),sep=",")
# View(reviews_text)
# write.table(reviews_text,file = "/home/sunil/Desktop/sentiment_analysis/Sentiment Analysis Dataset.csv",row.names = F,col.names = T,sep=',')



reviews_text <- read.csv("C:/Users/Mohammad opal/Downloads/ECE579_Project/text_emotion.csv")
View(reviews_text)
reviews_text_trim<-reviews_text[,-1]
reviews_text_trim<-reviews_text_trim[,-2]
View(reviews_text_trim)#Visualizing the data frame
dim(reviews_text_trim)
str(reviews_text_trim)


#removing everything except alphabetic and number from text or 2ND column
reviews_text_trim$content<-gsub("[^[:alnum:] ]","",reviews_text_trim$content)
reviews_text_trim$content<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$", "", reviews_text_trim$content, perl=TRUE)

#data frame without column 1,3 & all unwanted character.
View(reviews_text_trim)
#writing the output to a file that can be consumed to in other project
write.table(reviews_text_trim,file = "c:/Users/Mohammad opal/Downloads/ECE579_Project/text_emotion_Dataframe.csv",row.names = F,col.names = T,sep=',')


#TRANSSFORMATION of 2ND column (which is a text column, named as content)
#convert 2ND column of reviews_text_trim into VECTOR
#And remove stop words and punctuation from 2ND column.
library(SnowballC)
library(tm)
train_corp<-VCorpus(VectorSource(reviews_text_trim$content))
View(train_corp)
print(train_corp)
#creating document term matrix of 2ND column
dtm_train<-DocumentTermMatrix(train_corp,control = list(tolower=TRUE,removeNumbers=TRUE,
                                                        stopwords=TRUE,removePunctuation=TRUE,stemming=TRUE))
inspect(dtm_train)

#Removing Spers terms
dtm_train<-removeSparseTerms(dtm_train, 0.99)
inspect(dtm_train)
#Spliting Text data(2ND) into train and test
#spliting DTM data into train and test 
dtm_train_train<-dtm_train[1:15000, ]
dtm_train_test<-dtm_train[15001:20000, ]
###factor is not needed since my 1ST column is not numerical, already factor
#dtm_train_train_lebels<-as.factor(as.character(text[1:8000]$sentiment))
#dtm_train_test_lebels<-as.factor(as.character(text[8001:10000]$sentiment))
#str(reviews_text_trim)
dtm_train_train_lebels<-reviews_text_trim$sentiment[1:15000]
dtm_train_test_lebels<-reviews_text_trim$sentiment[15001:20000]


#convert all zero to N
cellconvert<-function(x){
  x<-ifelse(x>0, "Y","N")
}
#applying the function to rows in training and test datasets
dtm_train_train<-apply(dtm_train_train, MARGIN =2, FUN=cellconvert)
dtm_train_test<-apply(dtm_train_test, MARGIN = 2, FUN=cellconvert)

#training the naive bayes classifier on the training dtm
library(e1071)
nb_senti_classifier=naiveBayes(dtm_train_train, dtm_train_train_lebels)
#printing the summary of the model created
summary(nb_senti_classifier)

#making predictions on the test data dtm
nb_predicts<-predict(nb_senti_classifier, dtm_train_test, type="class")
#printing the predictions from the model
print(nb_predicts)
#computing accuracy of the model 
library(rminer)
print(mmetric(nb_predicts, dtm_train_test_lebels, c("ACC")))
#dev.off()

