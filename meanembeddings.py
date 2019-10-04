

import numpy as np
#Not my original code but greatly appreciative of the code.
#Class allows for use with sklearn pipeline where a classifier needs a fit-transform functionality.

#Out-of-vocabulary words will be assigned a zero on unseen test data. In this test I have trained the embedding models on the entire corpus. In reality this is not practical unless
#the models have had access to enough vocabulary that aproximates the population. Classifiers are not shown the test sentences or labels but are benefiting from the word embeddings having been trained on the entire corpus.
#The word embeddings produced are the mean value across 200 dimensions. This 200 dimension size is manually set in the fasttext and word2vec models in the voter class.

class MeanEmbeddingVectorizer(object):
         
    def __init__(self, wordembedding):      
        self.wordembedding = wordembedding       
        self.dim = len(list(wordembedding.values())) 
         
    def fit(self, X, y):       
        return self

    def transform(self, X):    
        return np.array([
            np.mean([self.wordembedding[w] for w in words if w in self.wordembedding]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])