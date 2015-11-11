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


def reverseGeocode(lng,lat):
    #g =  geocoder.w3w([lat,lng],method='reverse',key='EBFZ9ETX')    
    #return '-'.join(g.json.get('words',''))
    g =  geocoder.google([lat,lng],method='reverse')    
    return g.city if g.city else g.state
    


#loadfiles('C:\Users\AravindKumarReddy\Downloads\SMMProject2')
def main():
    tweetdata = loadfiles('C:\Users\AravindKumarReddy\Downloads\SMMSample')
    tweetdata['w3wCode'] = map(reverseGeocode, tweetdata['lng'],tweetdata['lat'])
    
    print tweetdata
    
if __name__ == '__main__':
    main()
                
                
    