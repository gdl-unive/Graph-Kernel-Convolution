import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.utils as utils
from grakel.kernels import WeisfeilerLehman
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn


def max_comp(E, d):
    E = list(E)
    graph = csr_matrix((np.ones(len(E)), zip(*E)), [np.max(E) + 1] * 2)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    max_elms = np.argwhere(labels == unique[np.argmax(counts)])

    max_ed_list = [e for e in E if (e[0] in max_elms) and (e[1] in max_elms)]

    dnew = dict([((int(k), d[k])) for k in max_elms.flatten()])

    return max_ed_list, dnew


def one_hot_embedding(labels, nlabels):
    eye = torch.eye(nlabels)
    return eye[labels]


def get_egonets(x, edge_index, i, hops=2):
    fn, fe, _, _ = utils.k_hop_subgraph([i], num_hops=hops, edge_index=edge_index)
    node_map = torch.arange(fn.max() + 1)
    node_map[fn] = torch.arange(fn.shape[0])
    ego_edges = node_map[fe]
    ego_nodes = x[fn, :]
    return ego_nodes, ego_edges


class GKernel(nn.Module):
    def __init__(self, nodes, labels, filters=8, max_cc=False, hops=2):
        super(GKernel, self).__init__()

        self.gk = WeisfeilerLehman(n_iter=3, normalize=True)
        self.hops = hops

        self.P = nn.ParameterList()
        self.X = nn.ParameterList()
        self.Xp = nn.ParameterList()

        self.Padd = nn.ParameterList()
        self.Prem = nn.ParameterList()
        for fi in range(filters):
            self.Padd.append(nn.Parameter(torch.ones(nodes, nodes) * 0))
            self.Prem.append(nn.Parameter(torch.ones(nodes, nodes) * 0))

            self.P.append(nn.Parameter(torch.eye(nodes, nodes).float(), requires_grad=False))
            self.X.append(nn.Parameter(one_hot_embedding(torch.randint(labels, (nodes,)), labels), requires_grad=False))
            self.Xp.append(nn.Parameter(torch.zeros((nodes, labels)).float(), requires_grad=True))
        self.filters = filters
        self.store = [None] * filters

        self.nodes = nodes
        self.max_cc = max_cc

    def random_edit(self, i=None):
        if i is None:  # randomly select a filter
            self.i = torch.randint(self.filters, (1,))[0]
            i = self.i

        self.store[i] = (self.P[i].data.clone(), self.X[i].data.clone())
        self.last_edit = None

        if np.random.rand(1)[0] > 0.3:  # edit graph
            n_edits = torch.randint(3, (1,)) + 1
            n_edits = 1
            Pmat = self.P[i] * self.Prem[i].sigmoid().data + (1 - self.P[i]) * \
                self.Padd[i].sigmoid().data + 1e-8  # sample edits
            Pmat = Pmat / Pmat.sum()
            inds = np.random.choice(Pmat.shape[0]**2, size=(n_edits,), replace=False, p=Pmat.flatten().cpu().numpy(),)
            inds = torch.from_numpy(np.stack(np.unravel_index(inds, Pmat.shape), 0)).to(Pmat.device)

            inds = torch.cat([inds, inds[[1, 0], :]], -1)
            self.P[i].data[inds[0], inds[1]] = 1 - self.P[i].data[inds[0], inds[1]]
            self.last_edit = inds + 0

            if self.P[i].sum(-1).min() == 0:  # ensure no isolated nodes
                self.revert_edit()
                self.random_edit()
        else:  # edit labels
            n_edits = 1

            self.lab_ind = torch.randint(self.X[i].shape[0], (n_edits,))
            px = self.Xp[i].sigmoid().data
            PX = px / px.sum(-1, keepdims=True)
            self.lab_val = [np.random.choice(PX.shape[1], size=(1,), replace=False,
                                             p=PX[j, :].cpu().numpy(),) for j in self.lab_ind]

            self.X[i].data[self.lab_ind, :] = 0
            self.X[i].data[self.lab_ind, self.lab_val] = 1

    def backprop_edit(self, score):
        temp = 1e0
        i = self.i
        inds = self.last_edit
        with torch.no_grad():
            if self.last_edit is None:
                self.Xp[i].grad = -torch.ones_like(self.Xp[i]) * 0
                sigx = (self.Xp[i][self.lab_ind, self.lab_val] * temp).sigmoid()
                self.Xp[i].grad[self.lab_ind, self.lab_val] = -score * sigx * (1 - sigx)
                #####
                if score < 0:
                    self.revert_edit()
                return

            Pd = self.P[i].data[inds[0], inds[1]]

            dx = 2 * (Pd - 0.5) * score
            sigx = (self.Padd[i][inds[0], inds[1]] * temp).sigmoid()
            self.Padd[i].grad = torch.zeros_like(self.Padd[i])
            self.Padd[i].grad[inds[0], inds[1]] = -dx * sigx * (1 - sigx)

            sigx = (self.Prem[i][inds[0], inds[1]] * temp).sigmoid()
            self.Prem[i].grad = torch.zeros_like(self.Padd[i])
            self.Prem[i].grad[inds[0], inds[1]] = dx * sigx * (1 - sigx)

            if score < 0:
                self.revert_edit()

    def revert_edit(self, i=None):
        if i is None:
            i = self.i
        self.P[i].data, self.X[i].data = self.store[i]

    def forward(self, x, edge_index, not_used=None, fixedges=None):

        # graph similarity here
        convs = []
        egonets = [get_egonets(x, edge_index, i, self.hops) for i in torch.arange(x.shape[0])]
        def G1(i): return [set([(e[0], e[1]) for e in egonets[i][1].t().cpu().numpy()]),
                           dict(zip(range(egonets[i][0].shape[0]), egonets[i][0].argmax(-1).cpu().numpy()))]
        Gs1 = [G1(i) for i in range(x.shape[0])]

        if not self.max_cc:
            Gs2 = [[set([(e[0], e[1]) for e in torch_geometric.utils.dense_to_sparse(self.P[fi])[0].t().cpu().numpy()]),
                    dict(zip(range(self.nodes), self.X[fi].argmax(-1).flatten().detach().cpu().numpy()))] for fi in range(self.filters)]
        else:
            Gs2 = [max_comp(set([(e[0], e[1]) for e in torch_geometric.utils.dense_to_sparse(self.P[fi])[0].t().numpy()]),
                            dict(zip(range(self.nodes), self.X[fi].argmax(-1).flatten().detach().numpy()))) for fi in range(self.filters)]

        print("Computing kernel on %d egonets" % len(Gs1))
        sim = self.gk.fit_transform(Gs1 + Gs2)
        conv = torch.from_numpy(sim[:x.shape[0], -self.filters:])
        return conv.float().to(x.device)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self.reset = False

        self.lin = nn.Linear(embedding_dim, embedding_dim)
        self.codebook_init = False

    def reset_codebook(self, x):
        if self.codebook_init:
            centroid, label = kmeans2(x.detach().cpu().numpy(
            ), self._embedding.weight.detach().cpu().numpy(), minit='matrix')
        else:
            centroid, label = kmeans2((x + torch.randn_like(x) * 1e-4).detach().cpu().numpy(), self._num_embeddings)
        self._embedding.weight.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        if self.training:
            self.reset_codebook(flat_input)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized_ind = encoding_indices.view(input_shape[:-1])
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), quantized_ind, perplexity, encodings
