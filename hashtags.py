# -*- coding: utf-8 -*-
import pandas as pd
import re
import sys


def extractHashtags(path,path2):
    pattern = re.compile(u'([#@]\S+)')  
    data = pd.read_csv(path,
                       names=['text1','lat','lng'],header=None,dtype=str)
                   
    data['text1'] = data['text1'].apply(lambda x: ' '.join([i.lower() for i in re.findall(pattern,str(x))]))
    
    tweetdata = pd.read_csv(path2,
                            header=0,dtype=str, names = ['text','lat','lng','class','latcntr','lngcntr']) 
    
    tweetdata = pd.merge(tweetdata,data, how='inner',on=['lat','lng'])
    tweetdata['text'] = tweetdata['text'] +' '+ tweetdata['text1']  
    tweetdata = tweetdata.drop('text1',1)
    tweetdata.to_csv('final_withhashtags.csv',index=False,header=None)
	
if __name__ == '__main__':
	try:
		if(len(sys.argv) != 3):
			print 'Usage \'loadfiles.py folderpath\''
			sys.exit(-1)
		else:
			extractHashtags(sys.argv[1],sys.argv[2])
	except:
		print 'Exception'