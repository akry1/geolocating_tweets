import pandas as pd
import re
import sys

def removeSpecialChars(path):
    tweetdata = pd.read_csv(path,header=0,dtype=str, names = ['text','lat','lng','class','latcntr','lngcntr'])
    pattern = '[^A-Za-z0-9\s]'
    tweetdata['text'] = tweetdata['text'].apply(lambda x: re.sub(pattern,'',x))
    
    tweetdata.to_csv('final_withoutspecialchars3.csv',index=False,header=None)

if __name__ == '__main__':
	try:
		if(len(sys.argv) != 2):
			print 'Usage \'loadfiles.py folderpath\''
			sys.exit(-1)
		else:
			removeSpecialChars(sys.argv[1])
	except:
		print 'Exception'