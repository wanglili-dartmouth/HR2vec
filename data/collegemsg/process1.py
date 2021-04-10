import networkx as nx
graph = nx.read_weighted_edgelist("edgelist.tsv", delimiter="\t", nodetype=int,create_using=nx.MultiGraph())

#graph=nx.convert_node_labels_to_integers(graph,first_label=0)

print(len(graph.nodes()))
print(len(graph.edges()))
graph2=nx.Graph()
for u,v,data in graph.edges(data=True):
    w = data['weight'] if 'weight' in data else 1.0
    if graph2.has_edge(u,v):
        graph2[u][v]['weight'] = min(w,graph2[u][v]['weight'])
    else:
        graph2.add_edge(u, v, weight=w)
print(len(graph2.nodes()))
print(len(graph2.edges()))

nx.write_edgelist(graph2, "edgelist1.tsv", delimiter="\t", data=["weight"])
