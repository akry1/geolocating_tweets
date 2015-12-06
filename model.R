library('tm')

content =read.csv("C:\\Users\\AravindKumarReddy\\Downloads\\SMMSample\\final2.csv", stringsAsFactors=FALSE,header=FALSE)

data = content[c(1:10000),]

corpus = Corpus(VectorSource(data$V1))
corpus = tm_map(corpus,removeWords,stopwords("english"))
#corpus = tm_map(corpus,stemDocument)
dtm = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(dtm,0.995)
wordsDataFrame = as.data.frame(as.matrix(sparse))
#colnames(wordsAdded) = paste("A", colnames(wordsAdded))
wordsDataFrame$... = NULL
wordsDataFrame$V4 = data$V4

library('caTools')
set.seed(1)
split = sample.split(wordsDataFrame$ada,SplitRatio=0.7)
train = subset(wordsDataFrame,split ==TRUE)
test = subset(wordsDataFrame,split ==FALSE)

# library('rpart')
# cart1 = rpart(V4~., data = train, method ="class")
# predCART = predict(cart1,newdata=test)
# table(test$class,predCART[,2]>0.5)
# prp(cart1)

#Naive Bayes

library('e1071')

nb = naiveBayes(V4~.,data=train)
preds = predict(nb,newdata=test[-test$V4])
