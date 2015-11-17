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
import re

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
                            js = json.loads(line)
                            tweetdata.append([js['text'],js['location']['lat'],js['location']['lng']])                            
            else:
                directories.append(current)
    return  pd.DataFrame(tweetdata,columns=['text','lat','lng'])


def reverseGeocode(lat,lng):
    #g =  geocoder.w3w([lat,lng],method='reverse',key='EBFZ9ETX')    
    #return '-'.join(g.json.get('words',''))
    g =  geocoder.google([lat,lng],method='reverse')    
    return g.city if g.city else g.state
    
    
def mapLocation(lat,lng):
    def convert(x):
        x = int(x)
        diff = abs(x%5)
#        if  diff > 5:
#            x += 5 - diff  
#        else:
#            x -= diff
        return str(x-diff)
    return ':'.join([convert(lng),convert(lat)])
 
def loadStopWords(path):
    with open(path,'r') as file:
        return {i.replace('\n','').strip():1 for i in list(file)}
    
wordLocDict = {} 
mappingList = wordLocDict.items()
stopWords = loadStopWords('stopwords.csv')

def wordsForChiFeatures(text,loc):
    notAlphaNumeric = u'[^a-z0-9]'
    for i in text.split(' '):
        i= i.encode('ascii','xmlcharrefreplace').lower()
        if len(i)>3 :
            if not re.search(notAlphaNumeric,i) and i not in stopWords:
                wordLocDict[(i,loc)] = wordLocDict.get((i,loc),0) + 1
                
                
def assignFeature(text, location,feature,N):    
    if feature in text.split(' '):
        owc = wordLocDict.get((feature,location)) 
        ownotc = sum( j for i,j in wordLocDict.items() if i[0]==feature and i[1]!=location)
        onotwc = sum( j for i,j in wordLocDict.items() if i[0]!=feature and i[1]==location)
        return (owc+ownotc)*(owc*onotwc)*(1/float(N)) 
    else:
        return 0
                
def main(path):
    tweetdata = loadfiles(path)
    #tweetdata['location'] = map(reverseGeocode, tweetdata['lat'],tweetdata['lng'])
    tweetdata['location'] = map(mapLocation, tweetdata['lat'],tweetdata['lng'])    
    map(wordsForChiFeatures,tweetdata['text'], tweetdata['location'])    
    totalCount = sum(j for j in wordLocDict.values() if j>1)
    for i,j in wordLocDict.items():
        # change 1 to any value as per requirement
        if j>1 :
            tweetdata[str(i)] = map(lambda x,y:assignFeature(x,y,i[0],totalCount),tweetdata['text'],tweetdata['location'])
    tweetdata.to_csv('liw.csv',header=True, index=False,encoding='utf-8')
if __name__ == '__main__':
    #main('C:\Users\AravindKumarReddy\Downloads\SMMProject2')
    main('C:\Users\AravindKumarReddy\Downloads\SMMSample')
                
                
    