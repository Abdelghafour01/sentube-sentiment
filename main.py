import os, json, enchant, nltk, random

from pprint import pprint
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import TweetTokenizer, PunktSentenceTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem.snowball import SnowballStemmer

directory = os.fsencode(os.getcwd() + "/sentube") # Current working directory

# this function reads comment data from the json files, tokenizes them, removes stop words, and writes them in comments.json
def tokenize():

	fileCount = 0 # serves as an iterating index between files
	print("Retrieving data from sentube dataset...")

	dataset = [] # contains a list of lists, containing strings of comments
	videoTitles = [] # contains a list of strings for the title of the comments

	#This loop retrieves the comments and the title of the video
	for file in os.listdir(directory):
	    filename = os.fsdecode(file)
	    if filename.endswith("json"): 
	        data = json.load(open("sentube/" + filename))

	        # Collect a list of comments for each video.
	        dataset.append([])
	        videoTitles.append(data["title"])
	        for comment in data["comments"]:
	            dataset[fileCount].append(comment["text"])
	        
	        fileCount += 1

	os.system("cls")

	with open('rawData.json', 'w') as outfile:
	    json.dump(dataset, outfile, indent=4)
	    print("Raw Comments written in rawData.json")

	#At this stage, the data of the comments is retrieved from the JSON files, as a list of comments within a list of videos
	stop_words = set(stopwords.words("english"))
	toknizer = TweetTokenizer(reduce_len = True) #reduce_len parameter serves as a reducer for a long sequence of letters

	# For instance: "this is goooooood" becomes "this is goood" with a maximum of 3 letters
	tokenizedData = []

	#Tokenize the comments into words and filter stop words. Stop words are words that are very neutral
	videoCount = 0
	for video in dataset:
	    tokenizedData.append([])
	    for comment in video:
	    	token_comment = toknizer.tokenize(comment) # Tokenize the comments...
	    	filtered_sentence = [w for w in token_comment if not w in stop_words] # ... then remove stop words
	    	tokenizedData[videoCount].append(filtered_sentence)
	    videoCount += 1

	print("Comments Tokenized successfully!")
	# At This stage, the list tokenizedData contains data in the same format as the dataset, but it is tokenized
	# For instance: "This is a good comment" becomes ['this', 'is', 'a', 'good', 'comment']

	# print the Comments to comments.json
	with open('comments.json', 'w') as outfile:
	    json.dump(tokenizedData, outfile, indent=4)
	    print("Comments written in comments.json")

	#at this stage, the file comments.json contains tokenized comments without stop words.

# This function stems the tokens generated
def stem():
	dictionary = enchant.Dict("en_US") # Initialize the English dictionary
	stemmer = SnowballStemmer("english") # Initialize the English stemmer

	tokenizedData = json.load(open("comments.json")) # Open the tokenized list of comments from comments.json
	polishedData = []
	
	videoCount = 0
	for video in tokenizedData:
	    polishedData.append([]) # Initialize a list of Comments
	    
	    commentCount = 0
	    for comment in video:
	        polishedData[videoCount].append([]) # Initialize a list of tokens
	        
	        for i in range(len(comment)):
	            if dictionary.check(comment[i]) == True: # If the word exists in the English dictionary
	                polishedData[videoCount][commentCount].append(stemmer.stem(comment[i])) # Append the stem of the token to the comment
	                            
	        commentCount += 1
	    
	    if(videoCount % 5  == 0):
	        os.system("cls")
	        print("Stemming and Removing Irrelevant tokens: " + str(videoCount + 1) + " out of " + str(len(tokenizedData)) + " == " + str(round(videoCount / len(tokenizedData) * 100, 2)) + "% ...")

	    videoCount += 1

	os.system("cls")
	print("Data Polished and Stemmed 100% Done.")

	# print the stemmed Comments to stemComments.json
	with open('stemComments.json', 'w') as outfile:
	    json.dump(polishedData, outfile, indent=4)
	    print("Comments written in stemComments.json")

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
	document_words = set(document)
	features = {}
	
	for word in word_features:
		features['contains({})'.format(word)] = (word in document_words)
	
	return features

def naiveBayesClass():
	documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
	random.shuffle(documents)

	featuresets = [(document_features(d), c) for (d, c) in documents]
	train_set, test_set = featuresets[100:], featuresets[:100]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print(nltk.classify.accuracy(classifier, test_set))

	print(documents[0])
	
	classifier.show_most_informative_features(5)

def sentiment():
	sentiments = []
	sid = SentimentIntensityAnalyzer()
	stemmedData = json.load(open("stemComments.json"))
	for video in stemmedData:
	    videoSentiments = []
	    finalSentiments = {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
	    for comment in video:
	        #print(comment)
	        ss = sid.polarity_scores('\n'.join(comment))

	        videoSentiments.append(ss) 
	        for k in ss:
	            finalSentiments[k] += ss[k]
	        
	    
	    for k in finalSentiments:
	        finalSentiments[k] /= len(videoSentiments)

	    sentiments.append(finalSentiments)

	# print the stemmed data to stemmedOut.txt
	with open('stemmedOut.json', 'w') as outfile:
	    json.dump(sentiments, outfile, indent=4)

def main():
	tokenize()
	stem()
	sentiment()

main()