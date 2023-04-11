from multi_rake import Rake
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

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

rake = Rake()
import pandas as pd
df= pd.read_csv('data.csv')
df.insert(0,"index",0)
df.insert(3, "keywords", "covid19")
text = df['question']+" "+df['answer']
cnt=len(text)

for i in range(cnt):
    
    string = preprocess(text[i])
    words = rake.apply(string)
    cnt1=len(words)
    keywords=[words[j][0] for j in range(cnt1)]
    keywords=[word.split(" ") for word in keywords]
    keywords = sum(keywords, [])
    df["index"][i]=i
    df["keywords"][i]=set(keywords)
    
def createInvertedIndices(df):
    numEntries = df.shape[0]
    invertInd = {}
    
    for i in range (numEntries):
        entry = df.iloc[i]
        cord_uid = entry['index']    
        keywords = entry['keywords']
        for k in keywords:
            if k not in invertInd:
                invertInd[k] = []
                invertInd[k].append(cord_uid)
            else:
                invertInd[k].append(cord_uid)
    return invertInd
df.to_csv('data_withKeywords.csv')
invertedIndices = createInvertedIndices(df)
pickle.dump( invertedIndices, open( "invertedIndices_FINAL.p", "wb" ) )