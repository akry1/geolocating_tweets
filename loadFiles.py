# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:32:04 2015

@author: AravindKumarReddy
"""
import os
import os.path as op
import pandas as pd
import json
import geocoder
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

def loadfiles(path):
    directories = [path]
    #files = []
    tweetdata = []
    while len(directories)!=0:
        nextDir = directories.pop(0)
        for i in os.listdir(nextDir):
            current = op.join(nextDir,i)
            if op.isfile(current):
                ext = op.splitext(current)[-1].lower()
                if ext == '.json':
                    #files.append(current)
                    with open(current,'r') as f:
                        for line in f:
                            try:
                                js = json.loads(line)
                                tweetdata.append([js['text'].encode('ascii','ignore'),js['location']['lat'],js['location']['lng']])
                            except:
                                print 'Corrupt json:', current
                                pass
            else:
                directories.append(current)

    res =  pd.DataFrame(tweetdata,columns=['text','lat','lng'])
    res.to_csv('C:\Users\AravindKumarReddy\Downloads\SMMSample\\rawdata.csv',index=False,header=None)
    #return tweetdata


def reverseGeocode(lat,lng):
    #g =  geocoder.w3w([lat,lng],method='reverse',key='EBFZ9ETX')    
    #return '-'.join(g.json.get('words',''))
    g =  geocoder.google([lat,lng],method='reverse')    
    return g.city if g.city else g.state
    
    
def mapLocation(lat,lng):
    def convert(x):
        x = int(x)
        diff = abs(x%10)
#        if  diff > 5:
#            x += 10 - diff  
#        else:
#            x -= diff
        return str(x-diff)
    return ':'.join([convert(lng),convert(lat)])
 
def loadStopWords(path):
    with open(path,'r') as file:
        return {i.replace('\n','').strip():1 for i in list(file)}
    
wordLocDict = {} 
stopWords = loadStopWords('stopwords.csv')

def wordsForChiFeatures(text,loc):
    notAlphaNumeric = u'[^a-z0-9]'
    for i in text.split(' '):
        i= i.lower()
        if len(i)>3 :
            if not re.search(notAlphaNumeric,i) and i not in stopWords:
                wordLocDict[(i,loc)] = wordLocDict.get((i,loc),0) + 1
                
                
def assignFeature(text, location,feature,N):    
    if feature in text.split(' '):
        owc = wordLocDict.get((feature,location)) 
        ownotc = sum( j for i,j in wordLocDict.items() if i[0]==feature and i[1]!=location)
        onotwc = sum( j for i,j in wordLocDict.items() if i[0]!=feature and i[1]==location)
        return (owc+ownotc)*(owc+onotwc)*(1/float(N)) 
    else:
        return 0
                
def main(path):
    #tweetdata = loadfiles(path)
    tweetdata = pd.read_csv(path,header=0,dtype=str, names = ['text','lat','lng','class'])[:50000]
    
    #traindata, testdata = train_test_split(tweetdata,test_size=0.3, random_state=50)

    size = len(tweetdata)
    start = 7*size/10

    trainclass = tweetdata['class'][:start]
    testclass = tweetdata['class'][start:]

    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,
            stop_words='english',use_idf=True,encoding='utf-8',
            decode_error='ignore',lowercase=True)

    norm = Normalizer(copy=False)
    tfids = vectorizer.fit_transform(tweetdata['text'])
    normalized_tfids = norm.fit_transform(tfids)

    ch2 = SelectKBest(chi2, k=1000)
    #normalized_tfids = ch2.fit_transform(normalized_tfids,tweetdata['class'])

    data = pd.DataFrame(normalized_tfids.toarray())
    traindata = data[:start]
    testdata = data[start:]

    traindata = ch2.fit_transform(traindata,trainclass)
    testdata = ch2.fit_transform(testdata,testclass)

    #traindata= pd.DataFrame(traindata,columns=['text','lat','lng','class'])
    #testdata = pd.DataFrame(testdata,columns=['text','lat','lng','class'])
        
    
    #tweetdata['location'] = map(reverseGeocode, tweetdata['lat'],tweetdata['lng'])  
#    map(wordsForChiFeatures,tweetdata['text'], tweetdata['location'])    
#    totalCount = sum(j for j in wordLocDict.values() if j>1)
#    for i,j in wordLocDict.items():
#        # change 1 to any value as per requirement
#        if j>5 :
#            tweetdata[str(i)] = map(lambda x,y:assignFeature(x,y,i[0],totalCount),tweetdata['text'],tweetdata['location'])
#    tweetdata.to_csv('liw.csv',header=True, index=False,encoding='utf-8')
        
    #testdata= loadfiles('C:\Users\AravindKumarReddy\Downloads\SMMTest')
    #traindata['location'] = map(mapLocation, traindata['lat'],traindata['lng'])
    #testdata['location'] = map(mapLocation, testdata['lat'],testdata['lng'])
    #train_tfids =  vectorizer.fit_transform(traindata['text'])
    #test_tfids =  vectorizer.fit_transform(testdata['text'])

    #train_tfids = norm.fit_transform(train_tfids)
    #test_tfids = norm.fit_transform(test_tfids)
    #km = KMeans(n_clusters=2000, init='k-means++', max_iter=100, n_init=1)
    #km.fit(traindata[[1,2]])
    #y = traindata['class']

    nb = MultinomialNB(alpha=.1)
    nb.fit(traindata,trainclass)

    predictions = nb.predict(testdata)
    print predictions
    print '================================='
    print testclass
    print accuracy_score(testclass,predictions)
    
    #test labels
    #km.fit(testdata[[1,2]])
    
    #print accuracy_score(testdata['class'],predictions)
    #print accuracy_score(km.labels_,predictions)
    
    #pd.DataFrame( confusion_matrix(testdata['location'],predictions)).to_csv('confusion_mat.csv')
    
    #print predictions[1:200]
    
    
    
    
if __name__ == '__main__':
	try:
		if(len(sys.argv) != 2):
			print 'Usage \'P1-a.py folderpath\''
			sys.exit(-1)
		else:
			loadfiles(sys.argv[1])
	except:
		print ''
    
    
                
                
    