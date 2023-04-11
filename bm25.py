import os
import re
import urllib.request
import tarfile
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

#text preprocessing
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


def preprocess(text): 
    stop_words = set(stopwords.words("english"))
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    ##Stemming
    ps=PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stop_words]
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 
    text = " ".join(text) 
    
    return text

def bm25(articles, df_dic, title_w, abstract_w, query):
    corpus_title = []
    corpus_abstract = []
    
    for article in articles:
        arr = df_dic.iloc[article].to_numpy()
        #title
        if type(arr[1]) != float:
            preprocessedTitle = preprocess(arr[1])
            corpus_title.append(preprocessedTitle)
        else:
            corpus_title.append(" ")
        
        #abstract
        if type(arr[2]) != float:
            preprocessedAbst = preprocess(arr[2])
            corpus_abstract.append(preprocessedAbst)
        else:
            corpus_abstract.append(" ")
            
    query = preprocess(query)
    
    tokenized_query = query.split(" ")
    
    tokenized_corpus_title = [doc.split(" ") for doc in corpus_title]
    tokenized_corpus_abstract = [doc.split(" ") for doc in corpus_abstract]
    
    #running bm25 on titles
    bm25_title = BM25Okapi(tokenized_corpus_title)
    doc_scores_titles = bm25_title.get_scores(tokenized_query)
    #weighting array
    doc_scores_titles = np.array(doc_scores_titles)
    doc_scores_titles = doc_scores_titles**title_w
    
    #running bm25 on abstracts
    bm25_abstract = BM25Okapi(tokenized_corpus_abstract)
    doc_scores_abstracts = bm25_abstract.get_scores(tokenized_query)
    #weighting
    doc_scores_abstracts = np.array(doc_scores_abstracts)
    doc_scores_abstracts = doc_scores_abstracts ** abstract_w
    
    #summing up the two different scores
    doc_scores = np.add(doc_scores_abstracts,doc_scores_titles)
    
    #creating a dictionary with the scores
    score_dict = dict(zip(articles, doc_scores))
    
    #creating list of ranked documents high to low
    doc_ranking = sorted(score_dict, key=score_dict.get, reverse = True)
    
    #get top 100
    doc_ranking = doc_ranking[0:100]
    
    """for i in range(len(doc_ranking)):
        dic_entry = df_dic.get(doc_ranking[i])
        doc_ranking[i] = dic_entry[0]"""
    
    return doc_ranking

def getPotentialArticleSubset(query):
    #load in inverted indices
    invertedIndices = pickle.load(open("invertedIndices_FINAL.p", "rb"))
    
    #preprocess query and split into individual terms
    query = preprocess(query)
    queryTerms = query.split(' ')
    
    potentialArticles = []
    #concatenate list of potential articles by looping through potential articles for each word in query
    for word in queryTerms:
        if word in invertedIndices: #so if someone types in nonsensical query term that's not in invertedIndices, still won't break!
            someArticles = invertedIndices[word]
            potentialArticles = potentialArticles + someArticles
            
    #convert to set then back to list so there are no repeat articles
    potentialArticles = list(set(potentialArticles))
    return potentialArticles

def retrieve(queries):
    #performing information retrieval
    
    df_dic = pd.read_csv("data_withKeywords.csv")
    results = []
    for q in queries:
        articles = getPotentialArticleSubset(q)
        result = bm25(articles,df_dic,1,2,q)
        results.append(result)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    #Output results
    for query in range(len(results)):
        for rank in range(len(results[query])):
            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(results[query][rank]))
            text=df_dic["answer"][results[query][rank]]
            question=queries[query]
            question_answerer = pipeline("question-answering", model = model, tokenizer= tokenizer)

            print(question_answerer(question=question, context = text))
retrieve(["what should i do if i get covid19"])



