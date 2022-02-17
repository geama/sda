
import numpy as np # use numpy arrays
import statistics as stat
from anytree import Node, RenderTree, NodeMixin
from pyparsing import condition_as_parse_action

# define a set of suitable impurity functions (depending on the type of response)
class Impurity:
    def __init__(self,name:str):
        self.name = name
    def get_impurity(self,array:list,type):
        if self.type == "MSE":
            return stat.variance(array)
        elif self.type=='GINI':
            # (Warning: This is a concise implementation, but it is O(n**2)
            # in time and memory, where n = len(x).  *Don't* pass in huge
            # samples!)
            # Mean absolute difference
            mad = np.abs(np.subtract.outer(array, array)).mean()
            # Relative mean absolute difference
            rmad = mad/np.mean(array)
            # Gini coefficient
            g = 0.5 * rmad
            return g
        elif self.type=='TAU':
            c = array.count(mode(array))
            t = len(array)
            return ((c/t)**2)*t

# define base class for nodes
class MyBaseClass(object):  # Just a basic base class
    value = None            # it only brings the node value

# define node class by making it a generic binary tree node class
class MyNodeClass(MyBaseClass, NodeMixin):  # Add Node feature
    def __init__(self, name, indexes, impurity:Impurity, split=None, parent=None, children=None):
        super(MyNodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        self.impurity = impurity          # vue in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        if children:
             self.children = children
    

    def get_name(self):
        return self.name
    

    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        if var_name in feat:         #is_numeric(var) :      # split for numerical variables
            var = features[var_name]    # obtains the var column by identifiying the feature name 
            self.split = var_name + ">" + str(soglia) # compose the split string (just for numerical features)
            parent = self.name
            select = var[(self.indexes)] > soglia              # split cases belonging to the parent node
        elif  var_name in feat_nominal:         #is_numeric(var) :      # split for nominal variables
            var = feat_nominal[var_name]    # obtains the var column by identifiying the nominal feature name 
            self.split = var_name + " in " + str(soglia) # compose the split string (just for numerical features)
            parent = self.name
            select = np.array([i in soglia for i in var[(self.indexes)]]) # split cases belonging to the parent node
        else :
            print("Var name is not among the supplied features!")
            return
        
        left_i = self.indexes[~select]                      # to the left child criterion FALSE
        right_i = self.indexes[select]                      # to the right child criterion TRUE
        child_l = "n" + str(int(parent.replace("n",""))*2)
        child_r = "n" + str(int(parent.replace("n",""))*2 + 1)
        return MyNodeClass(child_l, left_i, None, parent = self), MyNodeClass(child_r, right_i, None, parent = self)   # instantiate left & right children
            
            
    # add a method to fast render the tree in ASCII
    def print_tree(self):
        for pre, _, node in RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.split, node.indexes)

# define a set of suitable impurity functions (depending on the type of response)

  
impurity  = Impurity('MSE')


print("Print a tree identified by its root", "\n")


# function for predicting an unseen value (recursive implementation)
#from __future__ import annotations
def pred_x(node, x) -> MyNodeClass :
# node is the current search node (at the very beginning the root)
# x is a dictionary zip(featurenames,x)
    if node.is_leaf:
        print(node.name)
        print(type(node))
        return node #MyNodeClass(node.name, node.indexes, node.impurity, parent = node.parent)
    else:
        if eval(node.split, x):
            print(" dx") # for didactical purposes only
            pred_x(node.children[1], x) # go to the right child
        else:
            print(" sx") # for didactical purposes only
            pred_x(node.children[0], x) # go to the left child

        

new = (3, 23)
new_n = ("a", "y")
#d = dict(zip(features_names, new))
#dn = dict(zip(n_features_names, new_n))
#d.update(dn)
#pred_x(my_tree, d)


# First one should set the contraints for the tree growing:
grow_rules = {'min_cases_parent': 10,
              'min_cases_child': 5,
              'min_imp_gain': 10^-5
             }


from statistics import mean
from statistics import variance
#print(len(x1),len(x2))
#print(features_names)
#print(features)
#print(len(features['x1']))
#print(variance(y),y)

#categorical variables are stored here: n_features. names of the variables: n_features_names
#impurit = Impurity("MSE")
import pandas as pd

#d = dict(features, **n_features)  #merges the two dicts
#df = pd.DataFrame(data=d)         #creates the dataframe

#print(df)
import csv


#features=pd.array([])
#features=df.iloc[:,[0,1,2,3,4,5,6,7]]

#features_names=('x1','x2','x3','x4')
#features=df.iloc[:,[0,1]]
#features_names=list(df.columns)
#features_names=features_names[:2]
#features=df.iloc[:,[0,1]]
#features=df
#n_features=dict(df.iloc[:,[2,3]])

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


# start a tree structure by instantiating its root
my_tree = MyNodeClass('n1', indici, impurity, None) 





def node_search_split(node, impurity, features, features_names):
    
    impurities_1=[]
    between_variance=[]
    splits=[]
    variables=[]
    combinazioni=[]
    distinct_values=np.array([])
    t=0
 
    if len(node.indexes) >= grow_rules['min_cases_parent']:
        
        for var in n_features_names:
            #print('categoric')
            #for i in range(len(n_features[str(var)])):
            distinct_values=np.array([])
            distinct_values=np.append(distinct_values,np.unique(n_features[str(var)]))
            #distinct_values.flatten
            for i in range(1,len(distinct_values)):
                combinazioni.append(list(itertools.combinations(np.unique(n_features[str(var)]), i)))
                #print(combinazioni,'combinazioni') 
                
                    
            combinazioni=combinazioni[1:]
            #print(combinazioni)      
            for index in combinazioni: 
                for i in index:
                    #print(type(i))
                    #print('i:',i)    
                    stump = node.bin_split(features, n_features, str(var),i)
                    if y[stump[0].indexes].size <= 1:
                        #return None
                        #print("##############sono qui nodo 0#######")
                        impurities_1.append(0)
                        impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
                    elif y[stump[1].indexes].size <= 1:
                        #return None
                        #print("##############sono qui nodo1#######")
                        impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                        impurities_1.append(0)
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
                    elif y[stump[0].indexes].size > 1 and y[stump[1].indexes].size > 1:
                        impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                        impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
            combinazioni=[]
            distinct_values=np.array([])
            distinct_values=list(np.append(distinct_values,np.unique(n_features[str(var)])))
    
            for i in range(len(distinct_values)):
            
                #distinct_values[i]=(distinct_values[i])
                #print('qua',distinct_values)
                #print('qua2',distinct_values[i],type(distinct_values[i]))
                stump = node.bin_split(features, n_features, str(var),distinct_values[i])

                if y[stump[0].indexes].size <= 1:
                    #return None
                    impurities_1.append(0)
                    impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
                elif y[stump[1].indexes].size <= 1:
                    #return None
                    impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                    impurities_1.append(0)
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
                elif y[stump[0].indexes].size > 1 and y[stump[1].indexes].size > 1:
                    impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                    impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
            
                        
        for var in features_names:
            #print('numeric')
            for i in range(len(features[str(var)])):

                    stump = node.bin_split(features, n_features, str(var),features[str(var)][i])
                    if y[stump[0].indexes].size <= 1:
                        impurities_1.append(0)
                        impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(features[str(var)][i])
                        variables.append(str(var))
                        t+=2
                    elif y[stump[1].indexes].size <= 1:
                        impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                        impurities_1.append(0)
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(features[str(var)][i])
                        variables.append(str(var))
                        t+=2
                    elif y[stump[0].indexes].size > 1 and y[stump[1].indexes].size > 1:
                         impurities_1.append((mean(y[stump[0].indexes])**2)*len(y[stump[0].indexes]))
                         impurities_1.append((mean(y[stump[1].indexes])**2)*len(y[stump[1].indexes]))
                         between_variance.append(sum(impurities_1[t:]))
                         splits.append(features[str(var)][i])
                         variables.append(str(var))
                         t+=2
        

                    
        #print('splits',len(splits))
        #print('impurities_1',len(impurities_1))
        #print('between_variance',len(between_variance))
        #print('variables',len(variables))
        return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))]

    
               
        #return min(medie)
                #print(y[stump[0].indexes])
                #print(features[str(var)][i])
            
        
            #impurities_1=[]
                #print(my_tree.impurity)
                #impurities.append()
                #stump[0].print_tree()
                #stump[1].print_tree()
                #impurities.append(impurity(stump))
    
        
#node_search_split(my_tree,impurity,features,features_names)
'''
first_value,soglia=node_search_split(my_tree,impurity,features,features_names)
print((first_value,soglia))

splitsx,splitdx = my_tree.bin_split(features, n_features, str(first_value),soglia)

r,t=node_search_split(splitsx,impurity,features,features_names)
print((r,t))

z,w=node_search_split(splitdx,impurity,features,features_names)
print((z,w)) 


splitsx2,splitdx2 =splitdx.bin_split(features, n_features, str(z),w)

first_valuesx2,sogliasx2=node_search_split(splitsx2,impurity,features,features_names)
print(first_valuesx2,sogliasx2)

splitsx3,splitdx3 =splitsx2.bin_split(features, n_features, str(first_valuesx2),sogliasx2)
print()

d,g=node_search_split(splitsx3,impurity,features,features_names)
print(d,g)



first_valuedx2,sogliadx2=node_search_split(splitdx2,impurity,features,features_names)
print(first_value2,soglia2)


splitsx3,splitdx3 =splitdx2.bin_split(features, n_features, str(first_valuedx2),sogliadx2)
print()

e,s=node_search_split(splitsx3,impurity,features,features_names)
print(e,s)
'''

def stop_rule(impurity_father,impurity_child):
    if impurity_child > impurity_father:
        return True
    else:
        return False



class completetree:
    bigtree =  []
    devian_y = len(y)*variance(y)
    father = []
    node_prop_list = []

def growing_tree(node:Node,impurity,features,features_names,rout,prop=0.7):

    number_of_split = 0
    value_soglia_variance = []
    '''
    Salvare tutti i nodi dove è stato possibile effettuare lo split
    Inoltre tenere in considerazione sempre il root node e tutti gli split effettuati con successo perchè perchè nel momento in cui l'ultimo split avviene 
    quando si risale l'albero appena si arriva al root_node ci si ferma.
    '''
    #la condizione va su ko solo se ritorno al root node partendo da destra
    #father_variance = 1000000000000
    count_node = 0
    tree= [] 

    try:
        
        value,soglia,varian = node_search_split(node,impurity,features,features_names)                

    except TypeError:
        return None
    
    if rout == 'start':
        completetree.father.append(node)
        ex_deviance = varian - len(y)*mean(y)**2
 
    
    value_soglia_variance.append([value,soglia,varian])
    
    left_node,right_node = node.bin_split(features, n_features, str(value),soglia)
    

    tree.append((node,left_node,right_node))
    completetree.bigtree.append(node)
    completetree.father.append(node)
    completetree.bigtree.append(left_node)
    completetree.bigtree.append(right_node)
    print(value_soglia_variance,rout)
    print(tree)
    
    if rout == 'start':
        completetree.father.append(node)
        ex_deviance = varian - len(y)*mean(y)**2
    else:
        ex_deviance_list= []
        for inode in completetree.bigtree:
            if inode not in completetree.father:
                ex_deviance_list.append(len(y[inode.indexes])*(mean(y[inode.indexes])-mean(y))**2)
                #ex_deviance_list.append(0)
        ex_deviance = sum(ex_deviance_list)

    node_propotion = ex_deviance/ completetree.devian_y
    completetree.node_prop_list.append(node_propotion)
    print(node_propotion)
    if len(completetree.node_prop_list)>1:
        delta = completetree.node_prop_list[-1] - completetree.node_prop_list[-2]
        if delta < 0.03:#all utente 
            completetree.bigtree.pop()
            #completetree.node_prop_list.pop()
            return None
    if node_propotion >= prop: #My personal choise :Per non avere un albero enorme mettere questo limite mi sembra appropriato
        value_soglia_variance.append(None)
        completetree.bigtree.pop()
        return None
    
    return growing_tree(left_node,impurity,features,features_names,"left"),growing_tree(right_node,impurity,features,features_names,"right")


    
        
        #father_variance = varian
        



growing_tree(my_tree,impurity,features,features_names,"start")

'''
Per ogni nodo costruisci il mini albero:
    papà

figlio      figlio 
destro      sinistro


Se un figlio non c'è mettici None
Struttura
con [[Padre,figliodestro,figliosinisto]["""""""]["""""""""]]
'''
