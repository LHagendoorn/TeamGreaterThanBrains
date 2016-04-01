
import RandomForest

data = RandomForest.getData()
forest = RandomForest.trainForest(data['Xtrain'], data['Ytrain'])
RandomForest.createClassification(forest,data['Xtest'],data['XtestDF'])