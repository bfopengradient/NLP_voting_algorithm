 

import pandas as pd
from pandas import read_csv
from sklearn.utils import resample
 
#Class rebalances data. 
class prep:
   
    #Balances class if needed. Labels are balanced for this exercise.
    def balancedata(test,training):      
        training_1 = training[3000:6000]
        result = pd.concat([test,training_1], ignore_index=True)
         
        training_majority = result[result.Label_bin==0]
        training_minority = result[result.Label_bin==1]
         #Upsample minority class
        training_minority_upsampled = resample(training_minority, 
                                replace=True,      
                                 n_samples=100,  #resample problem sentences to total of 100 which is 3% of training/test data sets.  
                                 random_state=123)  
        #Combine majority class with upsampled minority class
        df_upsampled = pd.concat([training_majority, training_minority_upsampled])
        df_upsampled.shape
        result_1 = df_upsampled 
        return result_1






 


