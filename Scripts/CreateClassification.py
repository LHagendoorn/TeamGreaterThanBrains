'''Creates the classification file'''

import pandas as pd
import numpy as np

'''Creates the classification file
    - Ypred = array of shape [n_samples, n_outputs]
    - XtestDF = dataframe of Xtest data. Columns: business_id, F1, F2, F3 ...'''
def create(Ypred, XtestDF):

    #convert array Ypred [0 1 0 0 0 1] to list of strings ['2 6']
    predList = []
    for row in Ypred:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        ding = sep.join(indices)
        predList.append(ding)

    #create dataframe object containing business_ids and list of strings


    #save in csv file

    '''
    new_submission = pd.read_csv('../input/sample_submission.csv')
    for i in range(len(new_submission)):
        new_submission.at[i,'labels'] = str(np.random.choice(9,4,replace=False)).strip('[]')

    print (new_submission[0:5])

    # Write to submission file
    new_submission.to_csv('./random_guess.csv',index=False)
    '''
    return True