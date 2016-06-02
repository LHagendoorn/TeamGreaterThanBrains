import pandas as pd

correct_order = [4, 6, 2, 1, 8, 3]
current_order = [6, 4, 8, 3, 2, 1]

data = pd.DataFrame([[6,1],[4,2],[8,3],[3,4],[2,5],[1,6]])

print data

#sort data to have same order as labels
indices = [current_order.index(filename) for filename in correct_order]
test = data.reindex(indices)

print '-----'

print test