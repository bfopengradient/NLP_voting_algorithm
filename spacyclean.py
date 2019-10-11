#Spacy clean up
import string
import spacy
from spacy.lang.en import English
from spacy.lemmatizer import Lemmatizer
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import numpy as np

#Class uses spacy to parse data and then splits the cleaned data into test and training for the pipeline and voting algorithm.
class clean:
    
    def spacy_cleanup(dirty_data):
        nlp = spacy.load('en')
        parser = English() 
        corpus_list = []
        for line in dirty_data.split('\n'):
            article= []
            for w in nlp(line.lower()):
                if not w.is_punct:
                    w.string.strip()
                    article.append(w.lemma_)                     
            corpus_list.append(article)
        return corpus_list       

    #Produce a corpus for the word embedding models.
    def get_corpus(cleaned_training_corpus): 
        print(" All sentences parsed")            
        corpus = []
        for i in tqdm(range(len(cleaned_training_corpus))):
            for line in cleaned_training_corpus[i]:
                words = [x for x in line.split()]
                corpus.append(words)        
        return corpus
         
    #Split data into test and training for pipeline and voting classifier. 75/25 split. 
    def split_data(cleaned_training_corpus,labels):
        X_train, X_test, y_train, y_test = train_test_split(cleaned_training_corpus,labels, test_size=0.25, shuffle=True,random_state=1) 
        return  X_train, X_test, y_train, y_test