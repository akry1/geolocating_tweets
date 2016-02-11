# geolocating_tweets
This project was done as part of Social Media Mining course taken at ASU. The project is about predicting the user location based on the tweet posted by him. Please see the 'Tweet Based Geolocation.pdf' file for project details. 


<h3>Pre Processing</h3>  
1) Loading data from json to csv  
   Run ‘loadfiles.py path’, Give input location to the folder where the original data is present.  
2) Removing all non-alphanumeric characters   
   Run ‘remove_chars.csv filename’  
3) Extract hashtags from raw data and merge with the output of above step for different dataset  
Run ‘hashtags.py raw_filename merge_filename’  

<h3>Classification</h3>  
1) ‘nb_classifier.py’ has the implementation of tf-ids, Inverse cluster frequencies and Naïve Bayes classifier.  
It has different modes:  
•	Run ‘python nb_classifier.py filepath 2 1000’  for Inverse Cluster Frequency Model, 
change 1000 to any number to increase number of tweets to process  
•	Run ‘python nb_classifier.py filepath 1 1000’  for unique words model, change 1000 to any number 
to increase number of tweets to process  
•	Search for below line and Change False to True to enable cross validation  
	def main(path, n =1000, cv=False)  
‘python nb_classifier.py filepath 1 1000’  to run cross validation.  

