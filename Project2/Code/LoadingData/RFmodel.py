import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from Load import *
from Output import *

start_time = time.time()

# load train data
df_traindata_caf = load_traindata_caffefeatures()
df_traindata_lab = load_traindata_labels()

# Load test data
df_testdata_caf = load_testdata_caffefeatures()

print("--- load data: %s seconds ---" % round((time.time() - start_time),2))

x_train = df_traindata_caf
y_train = df_traindata_lab
x_test = df_testdata_caf

# Train model
rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)

print("--- train model: %s seconds ---" % round((time.time() - start_time),2))

# Predict
preds = rf.predict_proba(x_test)
predsdf = pd.DataFrame(preds)

print("--- prediction: %s seconds ---" % round((time.time() - start_time),2))

# Create ouput file
to_outputfile(predsdf,1,'RF')

print("--- generate output: %s seconds ---" % round((time.time() - start_time),2))

print "\n\n ... Done"