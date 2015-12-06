import pandas as pd
import re

def removeSpecialChars():
    tweetdata = pd.read_csv('C:\Users\AravindKumarReddy\Downloads\SMMSample\\final2.csv',header=0,dtype=str, names = ['text','lat','lng','class','latcntr','lngcntr'])
    pattern = '[^A-Za-z0-9\s]'
    tweetdata['text'] = tweetdata['text'].apply(lambda x: re.sub(pattern,'',x))
    
    tweetdata.to_csv('C:\Users\AravindKumarReddy\Downloads\SMMSample\\final_withoutspecialchars.csv',index=False,header=None)
    tweetdata[1:100000].to_csv('C:\Users\AravindKumarReddy\Downloads\SMMSample\\final_withoutspecialchars2.csv',index=False,header=None)

if __name__ == '__main__':
    removeSpecialChars()