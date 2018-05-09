import sys
import json
import os
import enchant
import nltk
import numpy as np
import logging
import re
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import logging

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from nltk.tokenize import TweetTokenizer # Tokenizer for the input comments
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import movie_reviews, stopwords, subjectivity

logging.basicConfig(level=logging.DEBUG)

dictionary = enchant.Dict("en_US") # English dictionary of words.
directory = os.fsencode(os.getcwd() + "/sentube") # Current working directory

dataset = [] # Contains lists of comments for each video
videoTitles = []

fileCount = 0 #serves as an iterating index between files
print("Retrieving data from sentube dataset...")
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

os.system("clear")

# print the Comments to comments.json
with open('comments.json', 'w') as outfile:
    json.dump(dataset, outfile, indent=4)

#At this stage, the data of the comments is retrieved from the JSON files, as a list of comments within a list of videos

toknizer = TweetTokenizer(reduce_len = True) #reduce_len parameter serves as a reducer for a long sequence of letters
# For instance: "this is goooooood" becomes "this is goood" with a maximum of 3 letters

tokenizedData = []

videoCount = 0
for video in dataset:
    tokenizedData.append([])
    for comment in video:
        tokenizedData[videoCount].append(toknizer.tokenize(comment))
    videoCount += 1

print("Comments Tokenized successfully!")
# At This stage, the list tokenizedData contains data in the same format as the dataset, but it is tokenized
# For instance: "This is a good comment" becomes ['this', 'is', 'a', 'good', 'comment']

polishedData = []
videoCount = 0
for video in tokenizedData:
    polishedData.append([])
    commentCount = 0
    for comment in video:
        polishedData[videoCount].append([])
        for i in range(len(comment)):
            if dictionary.check(comment[i]) == True:
                polishedData[videoCount][commentCount].append(comment[i])
                            
        commentCount += 1
    
    if(videoCount % 5  == 0):
        os.system("clear")
        print("Removing Irrelevant tokens: " + str(videoCount + 1) + " out of " + str(len(dataset)) + " == " + str(round(videoCount / len(dataset) * 100, 2)) + "% ...")            

    videoCount += 1

os.system("clear")
print("Data Polished. 100% Done.")

#At this stage, all the non-english words of the dictionary are removed.

stemmer = SnowballStemmer("english")

stemmedData = []

videoCount = 0
for video in polishedData:
    stemmedData.append([])
    for comment in video:
        stemmedData[videoCount].append([stemmer.stem(t) for t in comment])
                             
    if(videoCount % 5  == 0):
        os.system("clear")
        print("Stemming polished comments: " + str(videoCount + 1) + " out of " + str(len(dataset)) + " == " + str(round(videoCount / len(dataset) * 100, 2)) + "% ...")            

    videoCount += 1

os.system("clear")
print("Stemming Done: 100%")

print(stemmedData[0][0])

sid = SentimentIntensityAnalyzer()

n_instances = 10000

sentiments = []
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

train_subj_docs = subj_docs[:800]
test_subj_docs = subj_docs[800:1000]
train_obj_docs = obj_docs[:800]
test_obj_docs = obj_docs[800:1000]
training_docs = train_subj_docs + train_obj_docs
testing_docs = test_subj_docs + test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))



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

pprint("stemmedOut.json generated successfully!")
os.system("start stemmedOut.json")

sentiments = []
for video in polishedData:
    videoSentiments = []
    finalSentiments = {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
    for comment in video:
        #print(comment)
        ss = sid.polarity_scores(''.join(comment))

        videoSentiments.append(ss) 
        for k in ss:
            finalSentiments[k] += ss[k]
        
    
    for k in finalSentiments:
        finalSentiments[k] /= len(videoSentiments)

    sentiments.append(finalSentiments)

# print the polished (non-stemmed) data to polishedOut.txt
with open('polishedOut.json', 'w') as outfile:
    json.dump(sentiments, outfile, indent=4)

pprint("polishedOut.json generated successfully!")
os.system("start polishedOut.json")