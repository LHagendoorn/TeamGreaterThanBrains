import pandas as pd
from itertools import chain #to flatten lists

data = pd.DataFrame([[0,1,0],[4,0,0],[0,0,3],[0,0,4],[2,0,0],[0,1,0]])

print data
getallen = data.values
getallen = list(chain.from_iterable(getallen))

test = filter(lambda x: x!=0,getallen)

print '-----'

print test