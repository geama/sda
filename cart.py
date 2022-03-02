import itertools
import numpy as np # use numpy arraysfrom
from  statistics import mean,variance,mode
from anytree import Node, RenderTree, NodeMixin
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
        
# define base class for nodes
class MyBaseClass(object):  # Just a basic base class
    value = None            # it only brings the node value

class MyNodeClass(MyBaseClass, NodeMixin):  # Add Node feature
    def __init__(self, name, indexes, split=None, parent=None, children=None,node_level= 0):
        super(MyNodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        #self.impurity = impurity          # vue in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        self.node_level = node_level
        if children:
             self.children = children
    

    def get_name(self):
        return self.name
    
    def get_level(self):
        return self.node_level
    
    def set_features(self,features):
        self.features = features
    

    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        if var_name in feat:         #is_numeric(var) :      # split for numerical variables
            var = self.features[var_name]    # obtains the var column by identifiying the feature name 
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
            
            
    # add a method to fast render the tree in ASCII
    def print_tree(self):
        for pre, _, node in RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.split, node.indexes)


def impur(vector):
    return (mean(vector)**2)*len(vector)

grow_rules = {'min_cases_parent': 10,
              'min_cases_child': 5,
              'min_imp_gain': 10^-5
             }
class CART:
    
    bigtree =  []
    nsplit = 0
    father = []
    root = []
    tree = []
    node_prop_list = []

    def __init__(self,y,features,features_names,n_features,n_features_names):

        self.y = y
        self.features = features
        self.features_names = features_names
        self.n_features = n_features
        self.n_features_names = n_features_names
        self.devian_y = len(self.y)*variance(self.y)


    def node_search_split(self,node:MyNodeClass, features, features_names):
        
        impurities_1=[]
        between_variance=[]
        splits=[]
        variables=[]
        combinazioni=[]
        distinct_values=np.array([])
        t=0
        
        node.set_features(self.features)
    
        if len(node.indexes) >= grow_rules['min_cases_parent']:
            
            for var in self.n_features_names:
                #print('categoric')
                #for i in range(len(n_features[str(var)])):
                distinct_values=np.array([])
                distinct_values=np.append(distinct_values,np.unique(self.n_features[str(var)]))
                #distinct_values.flatten
                for i in range(1,len(distinct_values)):
                    combinazioni.append(list(itertools.combinations(np.unique(self.n_features[str(var)]), i)))
                    #print(combinazioni,'combinazioni') 
                    
                        
                combinazioni=combinazioni[1:]
                #print(combinazioni)      
                for index in combinazioni: 
                    for i in index:
                        #print(type(i))
                        #print('i:',i)    
                        stump = node.bin_split(self.features, self.n_features, str(var),i)
                        if self.y[stump[0].indexes].size <= 1:
                            #return None
                            #print("##############sono qui nodo 0#######")
                            impurities_1.append(0)
                            impurities_1.append(impur(self.y[stump[1].indexes]))
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(i)
                            variables.append(str(var))
                            t+=2
                        elif self.y[stump[1].indexes].size <= 1:
                            #return None
                            #print("##############sono qui nodo1#######")
                            impurities_1.append(impur(self.y[stump[0].indexes]))
                            impurities_1.append(0)
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(i)
                            variables.append(str(var))
                            t+=2
                        elif self.y[stump[0].indexes].size > 1 and self.y[stump[1].indexes].size > 1:
                            impurities_1.append(impur(self.y[stump[0].indexes]))
                            impurities_1.append(impur(self.y[stump[1].indexes]))
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(i)
                            variables.append(str(var))
                            t+=2
                combinazioni=[]
                distinct_values=np.array([])
                distinct_values=list(np.append(distinct_values,np.unique(self.n_features[str(var)])))
        
                for i in range(len(distinct_values)):
                
                    #distinct_values[i]=(distinct_values[i])
                    #print('qua',distinct_values)
                    #print('qua2',distinct_values[i],type(distinct_values[i]))
                    stump = node.bin_split(self.features, self.n_features, str(var),distinct_values[i])

                    if self.y[stump[0].indexes].size <= 1:
                        #return None
                        impurities_1.append(0)
                        impurities_1.append(impur(self.y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(distinct_values[i])
                        variables.append(str(var))
                        t+=2
                    elif self.y[stump[1].indexes].size <= 1:
                        #return None
                        impurities_1.append(impur(self.y[stump[0].indexes]))
                        impurities_1.append(0)
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(distinct_values[i])
                        variables.append(str(var))
                        t+=2
                    elif self.y[stump[0].indexes].size > 1 and self.y[stump[1].indexes].size > 1:
                        impurities_1.append(impur(self.y[stump[0].indexes]))
                        impurities_1.append(impur(self.y[stump[1].indexes]))
                        between_variance.append(sum(impurities_1[t:]))
                        splits.append(distinct_values[i])
                        variables.append(str(var))
                        t+=2
                
                            
            for var in self.features_names:
                #print('numeric')
                for i in range(len(self.features[str(var)])):

                        stump = node.bin_split(self.features, self.n_features, str(var),self.features[str(var)][i])
                        if self.y[stump[0].indexes].size <= 1:
                            impurities_1.append(0)
                            impurities_1.append(impur(self.y[stump[1].indexes]))
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(self.features[str(var)][i])
                            variables.append(str(var))
                            t+=2
                        elif self.y[stump[1].indexes].size <= 1:
                            impurities_1.append(impur(self.y[stump[0].indexes]))
                            impurities_1.append(0)
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(self.features[str(var)][i])
                            variables.append(str(var))
                            t+=2
                        elif self.y[stump[0].indexes].size > 1 and self.y[stump[1].indexes].size > 1:
                            impurities_1.append(impur(self.y[stump[0].indexes]))
                            impurities_1.append(impur(self.y[stump[1].indexes]))
                            between_variance.append(sum(impurities_1[t:]))
                            splits.append(self.features[str(var)][i])
                            variables.append(str(var))
                            t+=2
            

                        
            #print('splits',len(splits))
            #print('impurities_1',len(impurities_1))
            #print('between_variance',len(between_variance))
            #print('variables',len(variables))
            return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))]



    def growing_tree(self,node:Node,rout='start',prop=0.55,delta_check  = 0.1,node_proportion_partial_check = 0.9):
        
        value_soglia_variance = []

        mini_tree= [] 

        try:
            
            value,soglia,varian = self.node_search_split(node,self.features,self.features_names)                

        except TypeError:
            return None
        
        level = node.get_level()
        value_soglia_variance.append([value,soglia,varian,level])
        self.root.append((value_soglia_variance,rout))

        left_node,right_node = node.bin_split(self.features, self.n_features, str(value),soglia)



        mini_tree.append((node,left_node,right_node))
        self.tree.append(mini_tree) 
        self.bigtree.append(node)
        if rout != 'start':
            self.father.append(node)
        self.bigtree.append(node)
        self.bigtree.append(left_node)
        self.bigtree.append(right_node)
        print(value_soglia_variance,rout)

    ###### Calcolo della deviance nel nodo  

        if rout == 'start':
            self.father.append(node)
            ex_deviance = varian - len(self.y)*mean(self.y)**2
        else:
            ex_deviance_list= []
            for inode in self.bigtree:
                if inode not in self.father:
                    #print("inode figlio ", inode)
                    ex_deviance_list.append(len(self.y[inode.indexes])*(mean(self.y[inode.indexes])-mean(self.y))**2)
                    #ex_deviance_list.append(0)
            ex_deviance = sum(ex_deviance_list)

        node_propotion_total = ex_deviance/ self.devian_y   
        father_deviance = len(self.y[node.indexes])*variance(self.y[node.indexes])
        children_deviance = len(self.y[left_node.indexes])*variance(self.y[left_node.indexes]) + len(self.y[right_node.indexes])*variance(self.y[right_node.indexes])
        print("node_propotion_total ",node_propotion_total)
        node_propotion_partial = children_deviance/ father_deviance
        self.node_prop_list.append(node_propotion_total)
        print("Node propotion_partial ",node_propotion_partial)
        
        if len(self.node_prop_list)>1:
            delta = self.node_prop_list[-1] - self.node_prop_list[-2]
            print("delta ",delta)
            if delta < delta_check or node_propotion_partial > node_proportion_partial_check:#all utente  :Controllo delle variazione nei nodi figli
                #completetree.tree.pop()
                #completetree.root.pop()
                return None

        if node_propotion_total >= prop: 
            #completetree.tree.pop()
            #completetree.root.pop() 
            return None


        self.nsplit += 1
        return self.growing_tree(left_node,"left"),self.growing_tree(right_node,"right")

