import pickle
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import torch
import torch_geometric
import torch_geometric.utils as tut
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets.zinc import ZINC


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def clean_graph(x, edge_index):
    connected_nodes = set([i.item() for i in edge_index.flatten()])
    isolated_nodes = [i for i in range(x.shape[0]) if i not in connected_nodes]
    mask = torch.ones((x.shape[0],)).bool()
    mask[isolated_nodes] = False
    mapping = -torch.ones((x.shape[0],)).long()
    mapping[mask] = torch.arange(mask.sum())

    edge_index = mapping[edge_index]
    x = x[mask,:].int()
    return x, edge_index


class TUDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 hops: int = 2,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = True, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.hops=hops
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return []

    @property
    def processed_file_names(self) -> str:
        return  self.name + '_%dhops.pt' % self.hops

    def download(self):
        pass

    def process(self):
        graphs, num_classes = load_data(self.name, False)


        counter = 0
        data_list=[]
        for graph in graphs:
            X_concat = graph.node_features
            Adj_block = self.__preprocess_neighbors_sumavepool([graph])
            edge_index = Adj_block.coalesce().indices()

            X_concat,  edge_index= clean_graph(X_concat,edge_index)
            nidx = counter + torch.arange(X_concat.shape[0])
            counter = counter + X_concat.shape[0]

            egonets = [torch_geometric.utils.k_hop_subgraph([i], num_hops=self.hops, edge_index=edge_index,relabel_nodes=True)[:2] for i in range(X_concat.shape[0])]
            egonets_idx, egonets_edg = zip(*egonets)

            data_list.append(Data(x=X_concat, edge_index=edge_index, y=graph.label, nidx=nidx, egonets_idx=list(egonets_idx), nodes=X_concat.shape[0],egonets_edg=list(egonets_edg)))

        self.data, self.slices = self.collate(data_list)


        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block


class EgonetLoader(torch_geometric.data.DataLoader):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(data, **kwargs)


    def __iter__(self):

        for batch in super().__iter__():
            batch.egonets_idx = [x+batch.nodes[:i].sum()  for i,egonets_idx in enumerate(batch.egonets_idx) for x in egonets_idx]
            batch.egonets_edg = sum(batch.egonets_edg,[])
            yield batch


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

class ZINCDataset(InMemoryDataset):
    def __init__(self, root: str, name='ZINC',
                 hops: int = 2,
                 split='train',
                 transform = None,
                 pre_transform = None):

        self.hops=hops
        self.name = 'ZINC'
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.y = self.data.y[:,None]

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        return  self.name + '_%s_%dhops.pt' % (self.split,self.hops)

    def download(self):
        pass

    def process(self):
        ds = ZINC('./data_zinc',subset=True,split=self.split)

        counter=0
        data_list=[]
        for graph in ds:
            nidx = torch.arange(graph.x.shape[0]) + counter

            egonets = [torch_geometric.utils.k_hop_subgraph([i], num_hops=self.hops, edge_index=graph.edge_index,relabel_nodes=True)[:2] for i in range(graph.x.shape[0])]
            egonets_idx, egonets_edg = zip(*egonets)

            x = one_hot_embedding(graph.x[:,0],21)
            data_list.append(Data(x=x, edge_index=graph.edge_index, y=graph.y, nidx=nidx, egonets_idx=list(egonets_idx), nodes=graph.x.shape[0],egonets_edg=list(egonets_edg)))

        self.data, self.slices = ZINC.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        print(self.data)
        return f'{self.name}({len(self)})'


class StructuresDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, test=False, structure='grid', samples=200, hops=3):

        self.hops=hops
        with open("src/dataset/%s_dataset_size_10_20_p0.02.pkl" % structure,'rb') as f:
            D = pickle.load(f)
            self.Dpos = []
            self.Dneg = []

            self.motif = D[0]

            if train:
                self.Dpos += D[1][:int(samples*0.8)]
                self.Dneg += D[2][:int(samples*0.8)]

            if test:
                self.Dpos += D[1][int(samples*0.8):samples]
                self.Dneg += D[2][int(samples*0.8):samples]

        self.n_features = 1
        self.n_features = 1
        self.num_classes = 2


    def __len__(self):
        return len(self.Dpos)*2

    def __getitem__(self, idx):
        if idx%2:
            edge_index = torch.tensor(self.Dpos[idx//2]).long().t()
            y = torch.tensor(1)
        else:
            edge_index = torch.tensor(self.Dneg[idx//2]).long().t()
            y = torch.tensor(0)
        X = torch.ones(edge_index.max()+1).float()[:,None]
        edge_index = tut.to_undirected(edge_index)

        egonets = [torch_geometric.utils.k_hop_subgraph([i], num_hops=self.hops, edge_index=edge_index,relabel_nodes=True)[:2] for i in range(X.shape[0])]
        egonets_idx, egonets_edg = zip(*egonets)

        return Data(x=X, y=y, edge_index=edge_index,nidx=X[:,0]*0, egonets_idx=list(egonets_idx),nodes=X.shape[0],egonets_edg=list(egonets_edg))
