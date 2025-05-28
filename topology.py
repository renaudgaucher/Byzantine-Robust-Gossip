import networkx as nx
import tools
import numpy as np


class CommunicationNetwork(nx.Graph):

    """
    Communication Network of the Honest subgraph. The implementation here provides only
    gossip derived from (unitary-weighted) laplacian matrix of the graph. 
    """

    def __init__(self, incoming_graph_data, byz, weights_method="metropolis"):

        super().__init__(incoming_graph_data)
        self.byz=byz # number of Byzantines in the adjacency of each honest nodes.
        self.construct_weights(weights_method)


        self.spectrum = nx.laplacian_spectrum(self).astype(np.float32)
        self.laplacian = nx.laplacian_matrix(self).astype(np.float32).todense()
        self.adjacency_matrix = nx.adjacency_matrix(self).astype(np.float32).todense()
        self.algebraic_connectivity = self.spectrum[1]
        self.largest_eig = self.spectrum[-1]
        try:
            self.fiedler_vec = nx.fiedler_vector(self, method='tracemin_lu').astype(np.float32)
        except:
            print("Graph is not connected")
        self.communication_step =1/self.largest_eig
        


    def weights(self, j):
        # return the weights of associated with neighbors of node j, taken from the ajacency matrix. 
        
        res =  list(self.adjacency_matrix[j,:]) + [np.max(self.adjacency_matrix[j,:])]*self.byz
        if sum(res) < 1:
            res[j] = 1 - sum(res)
        return res
    
        # return [self.communication_step]*self.degree(j)+ [0] +  [self.communication_step]*f
    
    def construct_weights(self, weights_method):
        
        if weights_method=="metropolis":
            for e in self.edges:
                self.edges[e]['weight'] = 1/(max(self.degree[e[0]], self.degree[e[1]]) + 1 + self.byz)
        
        elif weights_method=="unitary":
            for e in self.edges:
                self.edges[e]['weight'] = 1


graph_types = ["fully_connected", "Erdos_Renyi", "lattice", "two_worlds", "random_geometric"]

def create_graph(name, size, hyper=None, byz=0, seed=None, *args, **kwargs):
    if name=="fully_connected":
        net = nx.complete_graph(size)
    elif name=="Erdos_Renyi":
        net = nx.erdos_renyi_graph(size, hyper, seed=seed)
    elif name=="lattice":
        net = nx.grid_graph(dim=[int(size**(1/hyper)) for i in range(hyper)], 
                                periodic=True)
    elif name=="two_worlds":
        c1 = nx.complete_graph(size//2)
        c2 = nx.complete_graph(size -size//2)
        c2 = nx.relabel_nodes(c2, {i:i+size//2 for i in range(size-size//2)}, copy=False)
        net = nx.union(c1,c2)

        for i in range(size//2):
            for k in range(int(hyper)):
                net.add_edge(i, (i+ k)%(size//2) + size//2)
    elif name=="random_geometric":
        net = nx.random_geometric_graph(size, radius=hyper, seed=seed, dim=2, p=2)
    else:
        raise ValueError(name + " is not a possible graph")
    
    return CommunicationNetwork(net, byz, *args, **kwargs)

# : Test that it works
import matplotlib.pyplot as plt
if __name__=="__main__":
    net = create_graph(name="two_worlds",size=10,hyper=2)
    print(f"algebraic connectivity: {net.algebraic_connectivity}; largest eig: {net.largest_eig}")
    nx.draw(net)
    plt.show()
    