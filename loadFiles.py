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
    
    
wordLocDict = {}    
def wordsForChiFeatures(text,loc):
    notAlphaNumeric = u'[^a-z0-9]'
    for i in text.split(' '):
        i= i.lower()
        if len(i)>3 :
            if not re.search(notAlphaNumeric,i):
                wordLocDict[(i,loc)] = wordLocDict.get((i,loc),0) + 1
                

#loadfiles('C:\Users\AravindKumarReddy\Downloads\SMMProject2')
def main():
    tweetdata = loadfiles('C:\Users\AravindKumarReddy\Downloads\SMMSample')
    #tweetdata['location'] = map(reverseGeocode, tweetdata['lat'],tweetdata['lng'])
    tweetdata['location'] = map(mapLocation, tweetdata['lat'],tweetdata['lng'])    

    map(wordsForChiFeatures,tweetdata['text'], tweetdata['location'])
    
    
    print { i:j for i,j in wordLocDict.items() if j>1 }
    
if __name__ == '__main__':
    main()
                
                
    