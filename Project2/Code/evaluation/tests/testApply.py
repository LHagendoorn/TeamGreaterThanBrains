import pandas as pd
import os

#get current directory
dir = os.path.dirname(__file__)

#read in submission file with probabilities
df = pd.read_csv(os.path.join(dir,'RandomForest.csv'))

df_filenames = df['img']
df_data = df.drop('img', axis=1)

print df_data.head()

replacer = lambda x: max(min(x,1-10**(-15)),10**(-15))

df2 = df_data.applymap(replacer)

print df2.head()