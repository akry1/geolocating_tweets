import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import sys

vectorizer = TfidfVectorizer(max_df=0.30,max_features=10000, use_idf=True,encoding='utf-8',
            decode_error='ignore')
norm = Normalizer(copy=False)
selector = SelectKBest(chi2, k='all')
nb = MultinomialNB(alpha=.1)

def main(path, n =1000, cv=False):
    tweetdata = pd.read_csv(path,header=0,dtype=str,names = ['text','lat','lng','class','latcntr','lngcntr'])[:n]

    centroids = { x[0][0]:(x[0][1],x[0][2]) for x in tweetdata.groupby(['class','latcntr','lngcntr'])}
    
    if not cv:
        traindata, testdata = train_test_split(tweetdata,test_size=0.3, random_state=50)
        predictions = classifier(traindata,testdata)
        print measureAccuracy(predictions,testdata[:,3],centroids)
        print measureAccuracy(predictions,testdata[:,3],centroids,1.5,5)
        print measureAccuracy(predictions,testdata[:,3],centroids,2.5,3)
        print measureAccuracy(predictions,testdata[:,3],centroids,5,7)
        print measureAccuracy(predictions,testdata[:,3],centroids,7.5,10)
    else:
        crossValidation(tweetdata,centroids)



def classifier(traindata,testdata,centroids=None):
    train_tfids = vectorizer.fit_transform(traindata[:,0])
    trainFinal = norm.fit_transform(train_tfids)
    trainFinal = selector.fit_transform(trainFinal,traindata[:,3])

    test_tfids = vectorizer.transform(testdata[:,0])
    testFinal = norm.fit_transform(test_tfids)
    testFinal = selector.transform(testFinal)

    nb.fit(trainFinal,traindata[:,3])

    predictions = nb.predict(testFinal)
    #print list(predictions)[:1000]
    #print '================================='
    #print list(testdata[:,3])[:1000]
    
    return predictions

def measureAccuracy(predictions,testclass, centroids, latDeviation=1.5, lngDeviation=3):
    if centroids:
        predLocations = [centroids.get(i,(-1000,-1000)) for i in predictions]
        testLocations = [centroids.get(i,(-1000,-1000)) for i in testclass]
        #print predLocations
        #print '\n++++++++++++++++++++++++++++'
        #print testLocations
        res = [1 if abs(float(predLocations[i][0])-float(testLocations[i][0]))<latDeviation and
                    abs(float(predLocations[i][1])-float(testLocations[i][1]))<lngDeviation else 0 for i in range(len(predLocations))]
        #print(classification_report([1 for i in range(len(predLocations))],res))
        return float(sum(res))/len(predLocations)


def crossValidation(data,centroids):
    tfids = vectorizer.fit_transform(data['text'])
    normalizedData = norm.fit_transform(tfids)
    finalData = selector.fit_transform(normalizedData,data['class'])
    
    def scorer(estimator,X,y):
        preds = estimator.predict(X)
        return measureAccuracy(preds,y,centroids,5,7)
        
    accuracy = cross_val_score(nb,finalData,data['class'],cv=10,scoring=scorer)
    print accuracy

def inverseClusterFreq(path, n=1000):
    #using custom classification
    tweetdata = pd.read_csv(path,header=0,dtype=str, names = ['text','lat','lng','class','latcntr','lngcntr'])[:n]
    traindata, testdata = train_test_split(tweetdata,test_size=0.3, random_state=50)
    clusters = pd.DataFrame(traindata,columns = ['text','lat','lng','class','latcntr','lngcntr']).groupby(['class','latcntr','lngcntr'])
    combined = []
    for group in clusters:
        lat = group[0][1]
        lng = group[0][1]
        text = ' '.join(list(group[1]['text']))
        combined.append([text,lat,lng,group[0][0]])
    centroids = { x[0][0]:(x[0][1],x[0][2]) for x in tweetdata.groupby(['class','latcntr','lngcntr'])}
    preds = classifier(np.array(combined),testdata)
    print measureAccuracy(preds,testdata[:,3],centroids,1.5,3)
    print measureAccuracy(preds,testdata[:,3],centroids,1.5,5)
    print measureAccuracy(preds,testdata[:,3],centroids,2.5,3)
    print measureAccuracy(preds,testdata[:,3],centroids,5,7)
    print measureAccuracy(preds,testdata[:,3],centroids,7.5,10)
    #crossValidation(pd.DataFrame(combined,columns = ['text','latcntr','lngcntr','class']),centroids)




#main('C:\\Users\\ayempada\\Downloads\\final_withoutspecialchars.csv')
if __name__ == '__main__':
    try:
        	if(len(sys.argv) != 4):
        		print 'Wrong Usage'
        		sys.exit(-1)
        	else:
        		tot= int(sys.argv[3].strip())
        		if sys.argv[2] == '1':
        			main(sys.argv[1],n=tot)
        		else:
        			inverseClusterFreq(sys.argv[1],n=tot)
    except:
        print 'exception'