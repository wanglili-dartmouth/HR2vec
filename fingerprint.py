import pandas as pd
from dateutil import parser
import networkx as nx
from scipy.stats import moment
from scipy.spatial import distance
import numpy as np
import math
from scipy import spatial
from sklearn.decomposition import IncrementalPCA,PCA
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

vector={}

sample = open('final_results.out', 'w') 
sns.set(style="whitegrid")
for name in ["barbell10_1","barbell10_2","barbell20_1","barbell20_2"]:
    vector[name]=[0,0]
for dim in [2]:#,4,10,40,80,128]:

    for mo in [4,6,8]:
        print("---------------------")
        for name in ["barbell10_1","barbell10_2","barbell20_1","barbell20_2"]:
            
            df = pd.read_csv(name+'.emb_'+str(dim),delimiter=',')
            
            #print(list(df.columns)[1])
            tmp=[]
            keys=list(df.columns)
            for i in range(1,len(keys)):
                tmp.append(df[keys[i]].tolist())
        
            vector[name]+=( np.array(moment(tmp, moment=mo) )  / ( np.var(tmp,axis=0)**(mo/2.0)  )  )* 2* ( (2/len(tmp))**(mo/2.0-1))
            print(vector[name])
            if(mo==8):
                plt.scatter(vector[name][0],vector[name][1],label=name)
            
        
        for name1 in ["barbell10_1","barbell10_2","barbell20_1","barbell20_2"]:
            for name2 in ["barbell10_1","barbell10_2","barbell20_1","barbell20_2"]:
                print("%.2f   " % spatial.distance.cosine(vector[name1], vector[name2]),end='', file = sample)
            print(file = sample)
        print("-----------------------------------",file = sample)
       
plt.legend()
plt.savefig("6barbell.pdf",bbox_inches='tight')
        
        