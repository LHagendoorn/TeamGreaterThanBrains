import numpy as np

data = [0.17,0.05,0.01,0.09,0.08,0.19,0.05,0.13,0.09,0.14]
data2 = [0.25,0.00,0.00,0.03,0.00,0.04,0.00,0.04,0.04,0.61]

#replace extremes
replacer = lambda x: max(min(x,1-10**(-15)),10**(-15))
data2 = map(replacer, data2)

#compute softmax data 2
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
data2 = [x*4 for x in data2]
data2 = softmax(data2)

#compute softmax data 1
data = [x*3 for x in data]
data = softmax(data)

print data
print data2


