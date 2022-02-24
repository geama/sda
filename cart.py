import itertools
import numpy as np # use numpy arraysfrom
from  statistics import mean,variance,mode
from anytree import Node, RenderTree, NodeMixin

# define a set of suitable impurity functions (depending on the type of response)
class Impurity:
    def __init__(self,name:str):
        self.name = name
    def get_impurity(self,array:list,type):
        if self.type == "MSE":
            return variance(array)
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

class feature:
    '''
    Classe necessaria contenente tutte le features
    '''

    def __init__(self,y,features,features_names,n_features_names,n_features) -> None:
        self.y = y
        self.features = features
        self.features_names = features_names
        self.n_features_names = n_features_names
        self.n_features = n_features

# define base class for nodes
class MyBaseClass(object):  # Just a basic base class
    value = None            # it only brings the node value

# define node class by making it a generic binary tree node class
class MyNodeClass(MyBaseClass, NodeMixin):  # Add Node feature
    def __init__(self, name, indexes, impurity:Impurity, split=None, parent=None, children=None,node_level= 0):
        super(MyNodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        self.impurity = impurity          # vue in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        self.node_level = node_level
        if children:
             self.children = children
    

    def get_name(self):
        return self.name
    
    def get_level(self):
        return self.node_level
    
    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        if var_name in feat:         #is_numeric(var) :      # split for numerical variables
            var = feature.features[var_name]    # obtains the var column by identifiying the feature name 
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
        return MyNodeClass(child_l, left_i, None, parent = self,node_level=self.node_level+1), MyNodeClass(child_r, right_i, None, parent = self,node_level=self.node_level+1)   # instantiate left & right children
          
grow_rules = {'min_cases_parent': 10,
              'min_cases_child': 5,
              'min_imp_gain': 10^-5
             }

class feature:
    '''
    Classe necessaria contenente tutte le features
    '''

    def __init__(self,y,features,features_names,n_features_names,n_features) -> None:
        self.y = y
        self.features = features
        self.features_names = features_names
        self.n_features_names = n_features_names
        self.n_features = n_features


def node_search_split(node, impurity, features, features_names):
    
    impurities_1=[]
    between_variance=[]
    splits=[]
    variables=[]
    combinazioni=[]
    distinct_values=np.array([])
    t=0
 
    if len(node.indexes) >= grow_rules['min_cases_parent']:
        
        for var in feature.n_features_names:
            #print('categoric')
            #for i in range(len(n_features[str(var)])):
            distinct_values=np.array([])
            distinct_values=np.append(distinct_values,np.unique(feature.n_features[str(var)]))
            #distinct_values.flatten
            for i in range(1,len(distinct_values)):
                combinazioni.append(list(itertools.combinations(np.unique(feature.n_features[str(var)]), i)))
                #print(combinazioni,'combinazioni') 
                
                    
            combinazioni=combinazioni[1:]
            #print(combinazioni)      
            for index in combinazioni: 
                for i in index:
                    #print(type(i))
                    #print('i:',i)    
                    stump = node.bin_split(features, feature.n_features, str(var),i)
                    if feature.y[stump[0].indexes].size <= 1:
                        #return None
                        #print("##############sono qui nodo 0#######")
                        impurities_1.append(0)
                        impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
                    elif feature.y[stump[1].indexes].size <= 1:
                        #return None
                        #print("##############sono qui nodo1#######")
                        impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                        impurities_1.append(0)
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
                    elif feature.y[stump[0].indexes].size > 1 and feature.y[stump[1].indexes].size > 1:
                        impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                        impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(i)
                        variables.append(str(var))
                        t+=2
            combinazioni=[]
            distinct_values=np.array([])
            distinct_values=list(np.append(distinct_values,np.unique(feature.n_features[str(var)])))
    
            for i in range(len(distinct_values)):
            
                #distinct_values[i]=(distinct_values[i])
                #print('qua',distinct_values)
                #print('qua2',distinct_values[i],type(distinct_values[i]))
                stump = node.bin_split(features, feature.n_features, str(var),distinct_values[i])

                if feature.y[stump[0].indexes].size <= 1:
                    #return None
                    impurities_1.append(0)
                    impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
                elif feature.y[stump[1].indexes].size <= 1:
                    #return None
                    impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                    impurities_1.append(0)
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
                elif feature.y[stump[0].indexes].size > 1 and feature.y[stump[1].indexes].size > 1:
                    impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                    impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                    between_variance.append(sum(impurities_1[t:]))
                    splits.append(distinct_values[i])
                    variables.append(str(var))
                    t+=2
            
                        
        for var in features_names:
            #print('numeric')
            for i in range(len(features[str(var)])):

                    stump = node.bin_split(features, feature.n_features, str(var),features[str(var)][i])
                    if feature.y[stump[0].indexes].size <= 1:
                        impurities_1.append(0)
                        impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(features[str(var)][i])
                        variables.append(str(var))
                        t+=2
                    elif feature.y[stump[1].indexes].size <= 1:
                        impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                        impurities_1.append(0)
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(features[str(var)][i])
                        variables.append(str(var))
                        t+=2
                    elif feature.y[stump[0].indexes].size > 1 and feature.y[stump[1].indexes].size > 1:
                         impurities_1.append((mean(feature.y[stump[0].indexes])**2)*len(feature.y[stump[0].indexes]))
                         impurities_1.append((mean(feature.y[stump[1].indexes])**2)*len(feature.y[stump[1].indexes]))
                         between_variance.append(sum(impurities_1[t:]))
                         splits.append(features[str(var)][i])
                         variables.append(str(var))
                         t+=2
        

        return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))]



class completetree:
    '''
    Classe per la costruzione dell'albero
    '''
    bigtree =  []
    devian_y = len(feature.y)*variance(feature.y)
    nsplit = 0
    father = []
    root = []
    Tree = []
    node_prop_list = []


def growing_tree(node:Node,impurity,features,features_names,rout='start',prop=0.55):
    
    value_soglia_variance = []

    mini_tree= [] 

    try:
        
        value,soglia,varian = node_search_split(node,impurity,features,features_names)                

    except TypeError:
        return None
    
    level = node.get_level()
    value_soglia_variance.append([value,soglia,varian,level])
    completetree.root.append((value_soglia_variance,rout))

    left_node,right_node = node.bin_split(features, feature.n_features, str(value),soglia)



    mini_tree.append((node,left_node,right_node))
    completetree.Tree.append(mini_tree) 
    completetree.bigtree.append(node)
    if rout != 'start':
        completetree.father.append(node)
    completetree.bigtree.append(node)
    completetree.bigtree.append(left_node)
    completetree.bigtree.append(right_node)
    print(value_soglia_variance,rout)

###### Calcolo della deviance nel nodo  

    if rout == 'start':
        completetree.father.append(node)
        ex_deviance = varian - len(feature.y)*mean(feature.y)**2
    else:
        ex_deviance_list= []
        for inode in completetree.bigtree:
            if inode not in completetree.father:
                #print("inode figlio ", inode)
                ex_deviance_list.append(len(feature.y[inode.indexes])*(mean(feature.y[inode.indexes])-mean(feature.y))**2)
                #ex_deviance_list.append(0)
        ex_deviance = sum(ex_deviance_list)

    node_propotion_total = ex_deviance/ completetree.devian_y   
    father_deviance = len(feature.y[node.indexes])*variance(feature.y[node.indexes])
    children_deviance = len(feature.y[left_node.indexes])*variance(feature.y[left_node.indexes]) + len(feature.y[right_node.indexes])*variance(feature.y[right_node.indexes])
    print("node_propotion_total ",node_propotion_total)
    node_propotion_partial = children_deviance/ father_deviance
    completetree.node_prop_list.append(node_propotion_total)
    print("Node propotion_partial ",node_propotion_partial)
    
    if len(completetree.node_prop_list)>1:
        delta = completetree.node_prop_list[-1] - completetree.node_prop_list[-2]
        print("delta ",delta)
        if delta < 0.015 or node_propotion_partial > 0.81:#all utente  :Controllo delle variazione nei nodi figli
            #completetree.tree.pop()
            #completetree.root.pop()
            return None

    if node_propotion_total >= prop: 
        #completetree.tree.pop()
        #completetree.root.pop() 
        return None


    completetree.nsplit += 1
    return growing_tree(left_node,impurity,features,features_names,"left"),growing_tree(right_node,impurity,features,features_names,"right")

