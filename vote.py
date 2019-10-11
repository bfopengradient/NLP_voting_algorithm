
 
import numpy as np
import pandas as pd
from Downloads.prepdata import prep 
from Downloads.spacyclean import clean
from gensim.models import FastText,word2vec
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from Downloads.meanembeddings  import MeanEmbeddingVectorizer
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
 

#Class 'voter' produces a confusion matrix for the voting classification algorithm.
class voter:
   
 	#Get fasttext embeddings
    def get_ft_embeddings(corpus):
        model = FastText(corpus,sg=0, hs=0, size=200, alpha=0.025, 
                     window=5, min_count=0, max_vocab_size=None, word_ngrams=1,
                     sample=0.001, seed=1, workers=4, min_alpha=0.0001, negative=5, 
                     cbow_mean=1, iter=5, null_word=0,
                     min_n=2, max_n=8, sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=300)         
        model.build_vocab(sentences=corpus,update=True)
        model.train(sentences=corpus,epochs=5,total_examples=model.corpus_count)
        ft_embeddings = dict(zip(model.wv.index2word, model.wv.vectors))
        return ft_embeddings

    #Get word2vec embeddings
    def get_w2v_embeddings(corpus):  
        model = word2vec.Word2Vec(corpus, alpha=0.025,sg=1,window=200,size=200,
                               min_count=0,workers=4,iter=5,seed=1,sample=0.01,batch_words=300,negative=5)
        model.build_vocab(sentences=corpus,update=True)
        model.train(sentences=corpus,epochs=5,total_examples=model.corpus_count)
        w2v_embeddings = dict(zip(model.wv.index2word, model.wv.vectors))
        return w2v_embeddings

   
    #Run a pipleine where the fasttext and woord2vec embeddings are each passed through a series of classifcation algorithms, which in turn are fed into
    #a voting classifier. The voting classifier in this case treats all classification algorithms equally in arriving at it's prediction.    
    def run_pipe(ft_embeddings,w2v_embeddings,X_train, X_test, y_train, y_test):
        #knn classifier
        knn=KNeighborsClassifier(n_neighbors=1)
        
        #Fasttex pipeline
        #KNN classifier,neighbours set low to one neighbour(will likely overfit)
        knn_ft = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(ft_embeddings)), 
        ("knn_clf",knn)])
       #Bagging classifier_base classifier knn    
        bag_ft = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(ft_embeddings)), 
        ("bag_clf", BaggingClassifier(base_estimator=knn, n_estimators=3, max_samples=.5, max_features=.5, 
                  bootstrap=True, bootstrap_features=False,
                  oob_score=False, warm_start=False, n_jobs=1, random_state=1, verbose=0))])
        #sklearn neural network classifier
        mlp_ft = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(ft_embeddings)), 
        ( "mlp_clf", MLPClassifier(solver='adam', alpha=1e-5,
                  hidden_layer_sizes=(15,),max_iter=2000, random_state=1))])        
        #Extratrees classifier
        et_ft = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(ft_embeddings)), 
        ("extra_trees_clf", ExtraTreesClassifier(n_estimators=10, max_features= 10,random_state=1,class_weight='balanced'))]) 
        
        #Word2vec pipeline
        #knn classifier,neighbours set low to one neighbour(will likely overfit)
        knn_w2v = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(w2v_embeddings)), 
        ("knn_clf",KNeighborsClassifier(n_neighbors=1))])  
        #Bagging classifier_base classifier knn    
        bag_w2v = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(w2v_embeddings)), 
        ("bag_clf", BaggingClassifier(base_estimator=knn, n_estimators=3, max_samples=.5, max_features=.5, 
                  bootstrap=True, bootstrap_features=False,
                  oob_score=False, warm_start=False, n_jobs=1, random_state=1, verbose=0))])    
        #sklearn neural network classifier
        mlp_w2v = Pipeline([
        ("embedding vectorizer", MeanEmbeddingVectorizer(w2v_embeddings)), 
        ( "mlp_clf", MLPClassifier(solver='adam', alpha=1e-5,
                  hidden_layer_sizes=(15,),max_iter=2000, random_state=1))])             
        #Extratrees classifier
        et_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_embeddings)), 
        ("extra_trees_clf", ExtraTreesClassifier(n_estimators=10, max_features= 10,random_state=1,class_weight='balanced'))])      

        #Voting Algorithm, for ease all classifiers are given equal weighting.
        total_vote =  VotingClassifier(estimators=[('knn_ft', knn_ft),('knn_w2v', knn_w2v),('bag_w2v',bag_w2v),('bag_ft',bag_ft),
                                              ('mlp_w2v',mlp_w2v),('mlp_ft',mlp_ft),('et_w2v',et_w2v),('et_ft',et_ft)],
                        voting='soft',flatten_transform=True,
                        weights=[1,1,1,1,1,1,1,1] ) 
        total_vote.fit(X_train,y_train)
        total_vote_at=total_vote.predict(X_test) 

        #Produce a confusion matrix for the Voting algorithm predictions. 
        conmat = np.array(confusion_matrix(y_test, total_vote_at))
        confusion = pd.DataFrame(conmat, index=['ok', 'problem'],
                         columns=['predicted ok','predicted problem'])  
        print('     VotingClassifier confusion matrix') 
        print('')
        print(confusion)
         

def main():
    test = pd.read_csv('/Users/brianfarrell/Desktop/test_client_dissatisfaction.csv',encoding='utf8')
    training = pd.read_csv('/Users/brianfarrell/Desktop/training.csv',encoding='utf8')
    
    data= prep.balancedata(test,training)
    cleaned_training_corpus = clean.spacy_cleanup(data.Sentences.str.cat(sep='\n'))
    corpus= clean.get_corpus(cleaned_training_corpus) 
    labels=data.Label_bin
    X_train, X_test, y_train, y_test = clean.split_data(cleaned_training_corpus,labels)
    ft_embeddings=voter.get_ft_embeddings(corpus) 
    w2v_embeddings=voter.get_w2v_embeddings(corpus)
    voter.run_pipe(ft_embeddings,w2v_embeddings,X_train, X_test, y_train, y_test)
 

if __name__ == "__main__":
    main()
 


         




         


         