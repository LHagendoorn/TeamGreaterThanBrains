import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from IO import Input
from IO import Output

start_time = time.time()

# load train data
df_trainset_caf = Input.load_trainset_caffefeatures()
df_trainset_lab = Input.load_trainset_labels()

# Load test data
df_validationset_caf = Input.load_validationset_caffefeatures()

print("--- load data: %s seconds ---" % round((time.time() - start_time),2))
start_time = time.time()

x_train = df_trainset_caf
y_train = df_trainset_lab
x_test = df_validationset_caf

# Train model
rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)

print("--- train model: %s seconds ---" % round((time.time() - start_time),2))
start_time = time.time()

# Predict
preds = rf.predict_proba(x_test)
predsdf = pd.DataFrame(preds)

print("--- prediction: %s seconds ---" % round((time.time() - start_time),2))
start_time = time.time()

# Create ouput file
Output.to_outputfile(predsdf,1,'RF')

print("--- generate output: %s seconds ---" % round((time.time() - start_time),2))

print "\n\n ... Done"
