
import RandomForest

data = RandomForest.getData()
forest = RandomForest.trainForest(data['Xtrain'], data['Ytrain'])
RandomForest.createProbabilities(forest,data['Xtest'])