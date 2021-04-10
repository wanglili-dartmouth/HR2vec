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
from CTDNE import CTDNE
from tqdm import tqdm
from CTDNE.edges import HadamardEmbedder
def split_edges(edges, graph,pivot_time):

    nodes_in_training=set()
    test_edges=[]
    train_edges=[]
    train_non_edges = []
    for edge in edges:
        if(edge[2]['time']>pivot_time):
            test_edges.append((edge[0],edge[1]))
        else:
            train_edges.append((edge[0],edge[1]))
            nodes_in_training.add(edge[0])
            nodes_in_training.add(edge[1])
    print("before ",len(test_edges))
    test_edges=[edge for edge in test_edges if (edge[0] in nodes_in_training and edge[1] in nodes_in_training)]
    print("after ",len(test_edges))
    random.seed(0)
    if(len(graph.nodes())<20000):
        non_edges = list(nx.non_edges(graph))
        
        used_non=random.sample(non_edges,len(test_edges)+len(train_edges))
        test_non_edges = used_non[:len(test_edges)]
        train_non_edges = used_non[len(test_edges):]
        
       
    else:
        print("Using another method")
        test_non_edges=[]
        train_non_edges=[]
        nodes = graph.nodes()
        N=len(test_edges)
        with tqdm(total=N, desc='False Edges', unit='false_edge') as pbar:
            while len(test_non_edges)<N:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph[random_edge[0]] and random_edge not in test_non_edges:
                    test_non_edges.append(random_edge)
                    pbar.update(1)
                    
        N=len(train_edges)
        with tqdm(total=N, desc='False Edges', unit='false_edge') as pbar:
            while len(train_non_edges)<N:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph[random_edge[0]] and random_edge not in train_non_edges and random_edge not in test_non_edges:
                    train_non_edges.append(random_edge)
                    pbar.update(1)
    print("check split edges:")
    print(len(train_non_edges))
    print(len(train_edges))
    print(len(test_non_edges))
    print(len(test_edges))



    return (train_edges, train_non_edges), (test_edges, test_non_edges)
def multigraph2graph(multi_graph_nx):
    '''
    convert a multi_graph into a graph, where a multi edge becomes a singe weighted edge
    Args:
        multi_graph_nx: networkx - the given multi_graph

    Returns:
        networkx graph
    '''
    if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
        print("No worries, No change")
        return multi_graph_nx
    graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

    if len(multi_graph_nx.nodes()) == 0:
        return graph_nx

    # add edges + attributes
    for u, v, data in multi_graph_nx.edges(data=True):
        data['weight'] = data['weight'] if 'weight' in data else 1.0

        if graph_nx.has_edge(u, v):
            graph_nx[u][v]['weight'] += data['weight']
        else:
            graph_nx.add_edge(u, v, **data)

    # add node attributes
    for node, attr in multi_graph_nx.nodes(data=True):
        if node not in graph_nx:
            continue
        graph_nx.nodes[node].update(attr)

    return graph_nx
def get_graph_T(graph_nx, min_time=-np.inf, max_time=np.inf, return_df=False):
    '''
    Given a graph with a time attribute for each edge, return the subgraph with only edges between an interval.
    Args:
        graph_nx: networkx - the given graph
        min_time: int - the minimum time step that is wanted. Default value -np.inf
        max_time: int - the maximum time step that is wanted. Default value np.inf
        return_df: bool - if True, return a DataFrame of the edges and attributes,
                          else, a networkx object

    Returns:
        sub_graph_nx: networkx - subgraph with only edges between min_time and max_time
    '''
    relevant_edges = []
    attr_keys = []

    if len(graph_nx.nodes()) == 0:
        return graph_nx

    for u, v, attr in graph_nx.edges(data=True):
        if min_time < attr['time'] and attr['time'] <= max_time:
            relevant_edges.append((u, v, *attr.values()))

            if attr_keys != [] and attr_keys != attr.keys():
                raise Exception('attribute keys in \'get_graph_T\' are different')
            attr_keys = attr.keys()

    graph_df = pd.DataFrame(relevant_edges, columns=['from', 'to', *attr_keys])

    if return_df:
        node2label = nx.get_node_attributes(graph_nx, 'label')
        if len(node2label) > 0:
            graph_df['from_class'] = graph_df['from'].map(lambda node: node2label[node])
            graph_df['to_class'] = graph_df['to'].map(lambda node: node2label[node])
        return graph_df
    else:
        sub_graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', list(attr_keys), create_using=type(graph_nx)())

        # add node attributes
        for node, attr in graph_nx.nodes(data=True):
            if node not in sub_graph_nx:
                continue
            sub_graph_nx.nodes[node].update(attr)

        return sub_graph_nx


def get_graph_times(graph_nx):
    '''
    Return all times in the graph edges attributes
    Args:
        graph_nx: networkx - the given graph

    Returns:
        list - ordered list of all times in the graph
    '''
    return np.sort(np.unique(list(nx.get_edge_attributes(graph_nx, 'time').values())))

def get_pivot_time(graph_nx, wanted_ratio=0.2, min_ratio=0.1):
    '''
    Given a graph with 'time' attribute for each edge, calculate the pivot time that gives
    a wanted ratio to the train and test edges
    Args:
        graph_nx: networkx - Graph
        wanted_ratio: float - number between 0 and 1 representing |test|/(|train|+|test|)
        min_ratio: float - number between 0 and 1 representing the minimum value of the expected ratio

    Returns:
        pivot_time: int - the time step that creates such deviation
    '''
    times = get_graph_times(graph_nx)
    if wanted_ratio == 0:
        return times[-1]

    time2dist_from_ratio = {}
    for time in times[int(len(times) / 3):]:
        train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=time))
        num_edges_train = len(train_graph_nx.edges())

        test_graph_nx = get_graph_T(graph_nx, min_time=time)
        print(time," before :",len(test_graph_nx.edges()))
        test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
        test_graph_nx = multigraph2graph(test_graph_nx)
        num_edges_test = len(test_graph_nx.edges())
        print(time," after :",len(test_graph_nx.edges()))
        
        current_ratio = num_edges_test / (num_edges_train + num_edges_test)
        print(time,"   ",current_ratio)
        if current_ratio <= min_ratio:
            continue

        time2dist_from_ratio[time] = np.abs(wanted_ratio - current_ratio)

    pivot_time = min(time2dist_from_ratio, key=time2dist_from_ratio.get)

    print(f'pivot time {pivot_time}, is close to the wanted ratio by {round(time2dist_from_ratio[pivot_time], 3)}')

    return pivot_time
def test(train_edges,embeddings):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    length=0
    for u,v in train_edges:
        train_x.append(embeddings[u]+embeddings[v])
        train_y.append(1)
        length=len(embeddings[u]+embeddings[v])
    for u,v in train_non_edges:
        train_x.append(embeddings[u]+embeddings[v])
        train_y.append(0)
    for u,v in test_edges:
        test_x.append(embeddings[u]+embeddings[v])
        test_y.append(1)
    for u,v in test_non_edges:
        test_x.append(embeddings[u]+embeddings[v])
        test_y.append(0)
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    roc=roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])
    print("Average: ",roc, file = sample)
    print("Feature length:",length, file = sample)
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    for u,v in train_edges:
        train_x.append(embeddings[u]*embeddings[v])
        length=len(embeddings[u]*embeddings[v])
        train_y.append(1)
    for u,v in train_non_edges:
        train_x.append(embeddings[u]*embeddings[v])
        train_y.append(0)
    for u,v in test_edges:
        test_x.append(embeddings[u]*embeddings[v])
        test_y.append(1)
    for u,v in test_non_edges:
        test_x.append(embeddings[u]*embeddings[v])
        test_y.append(0)
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    roc=roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])
    print("Hada: ", roc, file = sample)
    print("Feature length:",length, file = sample)
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    for u,v in train_edges:
        train_x.append(abs(embeddings[u]-embeddings[v]))
        train_y.append(1)
    for u,v in train_non_edges:
        train_x.append(abs(embeddings[u]-embeddings[v]))
        train_y.append(0)
    for u,v in test_edges:
        test_x.append(abs(embeddings[u]-embeddings[v]))
        test_y.append(1)
    for u,v in test_non_edges:
        test_x.append(abs(embeddings[u]-embeddings[v]))
        test_y.append(0)
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    roc=roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])
    print("L1: ", roc, file = sample)
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    for u,v in train_edges:
        train_x.append(abs(embeddings[u]-embeddings[v])*abs(embeddings[u]-embeddings[v]))
        train_y.append(1)
    for u,v in train_non_edges:
        train_x.append(abs(embeddings[u]-embeddings[v])*abs(embeddings[u]-embeddings[v]))
        train_y.append(0)
    for u,v in test_edges:
        test_x.append(abs(embeddings[u]-embeddings[v])*abs(embeddings[u]-embeddings[v]))
        test_y.append(1)
    for u,v in test_non_edges:
        test_x.append(abs(embeddings[u]-embeddings[v])*abs(embeddings[u]-embeddings[v]))
        test_y.append(0)
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    roc=roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])
    print("L2: ", roc, file = sample)
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0) 
    graph = nx.read_weighted_edgelist("data/collegemsg/edgelist.tsv", delimiter="\t", nodetype=None,create_using=nx.Graph())
    nx.set_edge_attributes(graph, name="time", values={edge: abs(weight) 
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
        
        

    pivot_time=get_pivot_time(graph, wanted_ratio=0.2, min_ratio=0.1)
    sample = open('collegemsg.out', 'w') 
    print(pivot_time, file = sample)
    edges = list(graph.edges.data())
    #non_edges = list(nx.non_edges(graph))

    (train_edges, train_non_edges), (test_edges, test_non_edges) = split_edges(edges, graph,pivot_time)

    graph.remove_edges_from( test_edges)
    #train_edges=list(graph.edges())
    #train_non_edges=random.sample(non_edges,len(train_edges)+len(test_edges))
    #test_non_edges=train_non_edges[len(train_edges):]
    #train_non_edges=train_non_edges[:len(train_edges)]
    print("train edges: ",len(train_edges))
    print("test edges: ",len(test_edges))
    print("train graph: ",len(list(graph.edges.data())))
    for experiment in [1]:

        model = HR2vec(graph.to_directed(), walk_length=10, num_walks=80,opt2_reduce_sim_calc=True,opt3_num_layers=1,workers=1,verbose=40 )
        model.train(embed_size=128,window_size = 5, iter = 3)
        embeddings = model.get_embeddings()
        print("HR2vec   ", file = sample)
        test(train_edges,embeddings)
        sample.flush()

        




