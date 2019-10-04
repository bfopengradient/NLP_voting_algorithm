#### NLP_voting_algorithm
 
Voting algorithm test.
Data is from a conduct risk domain. Sentences are labelled as problematic from a conduct risk perspective or innocent/neutral from a conduct risk perspective. For this exercise models are trained and tested on imbalanced classes. I have printed out the top of both the training and test data sets to give some context to the sentences used in the exercise.

#### There are four custom classes.
The "prep" class reads in the test and training data and rebalances the classes if required. In this exercise the class of interest is set at 3% of the entire dataset. It can be set at whatever is needed or just ignored if appropriate. The "clean" class uses Spacy to produce a corpus ready for the pipeline and ultimately the voting algorithm. The "mean embedding class" (not my original code) allows for the word embeddings to be passed to the sklearn pipeline. The "voter" class runs all the pipelines and produces a confusion matrix from the final voting algorithm.

#### Overview of the voting algorithm.
This is a natural language processing exercise. The data consists of sentences and accompanying labels. The final task performed in the notebook is to classify sentences that are regarded as ok or problematic. Problem sentences are those viewed as being of concern and would be viewed by a risk person as worthy of further examination. The confusion matrix summarises the accuracy of the voting algorithm. The various classes that are imported are doing the following under the hood:

Producing word embeddings from the Facebook Fasttext model and the Google Word2Vec model.
Each set of embeddings along with labels are passed to four classification algorithms:KNN,Bagging classifier(KNN base), fully connected neural network classifier and extratrees(extremely randomised trees) classifier. Each of these classification algorithms in effect produces a prediction for the voting algorithm but with their own bias-variance characteristics.
In total there are eight classifiers working in the pipeline ahead of the final voting algorithm. There are four classifiers for each of the two word embedding models.
The voting algorithm, in this exercise, treats each of of the eight predictions equally in deciding how to classify the unseen test sentences. Results/predictions are summarised in the confusion matrix.


#### Oct 2019

