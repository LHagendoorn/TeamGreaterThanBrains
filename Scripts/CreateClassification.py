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
    labelsDF = pd.DataFrame(predList, columns=['labels'])
    submission = pd.concat([XtestDF['business_id'], labelsDF], axis=1)

    #save in csv file
    submission.to_csv('./submission2016-03-09colorFeatures.csv',index=False)

    return True


'''Creates the probability file
    - Yprob = list with dimensions [n_outputs]x[n_samples]x[n_classes]'''
def createProbFile(Yprob):

    #transform 3d list into 2d list of size n_samples x n_outputs
    ff = np.array(Yprob)
    ff = np.delete(ff,0,2) #delete index 0 of 3rd dimension
    ff = np.squeeze(ff)    #remove singleton dimension
    ff = np.transpose(ff)

    #transform to dataframe
    probabilities = pd.DataFrame(ff)

    #save in csv file
    probabilities.to_csv('./probColor.csv',index=False, header=None)

    return True