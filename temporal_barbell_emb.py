from sklearn.metrics import roc_auc_score
import numpy as np
import random
from ge.classify import read_node_label, Classifier
from ge import *
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import json

if __name__ == "__main__":
  #  np.random.seed(0)
  #  random.seed(0) 
    for my_size in [2]:
        for name in ["barbell20_1","barbell20_2","barbell10_1","barbell10_2"]:
            graph = nx.read_weighted_edgelist(name+".tsv", delimiter=" ", nodetype=None,create_using=nx.Graph())
            nx.set_edge_attributes(graph, name="time", values={edge: abs(weight) 
                for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
            nx.set_edge_attributes(graph, name="weight", values={edge: 1
                for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
                
                
                
            model = HR2vec(graph.to_directed(), walk_length=10, num_walks=80,opt2_reduce_sim_calc=True,opt3_num_layers=1,workers=1,verbose=40 )
            model.train(embed_size=my_size,window_size = 5, iter = 3)
            embeddings = model.get_embeddings()
            data=pd.DataFrame(embeddings)
            data.to_csv(name+".emb_"+str(my_size))

    #    model =  Struc2Vec(graph.to_directed(), walk_length=10, num_walks=80,opt2_reduce_sim_calc=True,opt3_num_layers=1,workers=8,verbose=40 )
    #    model.train(embed_size=2,window_size = 5, iter = 3)
    #    embeddings = model.get_embeddings()
    #    data=pd.DataFrame(embeddings)
    #    data.to_csv(name+".struct2vec")
        
        
    #    model = Node2Vec(graph.to_directed(), walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 1)#init model
    #    model.train(embed_size=2,window_size = 5, iter = 3)
    #    embeddings = model.get_embeddings()
    #    data=pd.DataFrame(embeddings)
    #    data.to_csv(name+".node2vec")