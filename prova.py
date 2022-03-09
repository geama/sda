from cart import *

import pandas as pd

#d = dict(features, **n_features)  #merges the two dicts
#df = pd.DataFrame(data=d)         #creates the dataframe

#print(df)
import csv



#####################################loading the carseats data#########################

df=pd.read_csv('Carseats.csv')
df=df.iloc[:,1:]

features_names=list(df.columns)

colonne=features_names[:6]
features_name=features_names[7:9]
features_names=colonne + features_name




n_features_names=list(df.columns)
columns = [(n_features_names[6])]
n_features_name = n_features_names[9:11]
n_features_names=columns + n_features_name




features=df.iloc[:,0:6]
features2=df.iloc[:,7:9]

features=dict(features)
features2=dict(features2)


n_features=df.iloc[:,6:7]
n_features2=df.iloc[:,9:11]


n_features=dict(n_features)
n_features2=dict(n_features2)

features = dict(features, **features2)
n_features = dict(n_features, **n_features2)


#############################################################################


import itertools
from statistics import mode
'''
print(features,'features',type(features))
print()
print(features_names,'features_names')
print()
print(n_features,'n_features')
print()
print(n_features_names,'n_features_names')
print()
'''
########################### y categorical #######################################
High=[]
for i in features['Sales']:
    if i < 8:
        High.append('NO')
    else:
        High.append('YES')

High=pd.DataFrame(High)
High=dict(High)
High['High'] = High.pop(0)

#y=High['High']

#exclude_keys = ['Sales']

#new_d = {k: features[k] for k in set(list(features.keys())) - set(exclude_keys)}
#features=new_d

#features_names=features_names[1:]


######################y numerical#####################################
y=features['Price']
exclude_keys = ['Price']
new_d = {k: features[k] for k in set(list(features.keys())) - set(exclude_keys)}
features=new_d

indici = np.arange(0, len(y))

features_names3 = features_names[0:5]
features_names4 = features_names[6:]
features_names = features_names3 +  features_names4 

#impurity = impurity_fn('MSE') # chhose the simplest impurity functin (for regression tree)

impurity = Impurity ("MSE")
# start a tree structure by instantiating its root
my_tree = MyNodeClass('n1', indici, None) 


cart = CART(y,features,features_names,n_features,n_features_names) 

cart.growing_tree(my_tree)

print(len(cart.get_leaf()))
cart.pruning()


'''
def myfun():
    print("hello")

import dis
dis.dis(myfun)
'''