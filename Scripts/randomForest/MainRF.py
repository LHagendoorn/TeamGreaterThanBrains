
import RandomForest
import pandas as pd

data = RandomForest.getData()
forest = RandomForest.trainForest(data['Xtrain'], data['Ytrain'])
RandomForest.createProbabilities(forest,data['Xvalidation'])

#save business ids in csv file
busIds = pd.DataFrame(data['busIdsVal'])
busIds.to_csv('business_ids_validationset.csv',index=False, header=None)

#merge business ids with probabilities
a = pd.read_csv("probColorValidationSet.csv")
b = pd.read_csv("business_ids_validationset.csv")
merged = pd.concat([b,a], axis=1)
merged.to_csv("output.csv", index=False)