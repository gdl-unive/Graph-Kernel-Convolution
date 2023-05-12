from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from grakel.kernels import (GraphletSampling, Propagation, PyramidMatch,
                            RandomWalkLabeled, WeisfeilerLehman,
                            WeisfeilerLehmanOptimalAssignment)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)

from layers import *


def H(x): return -torch.sum(x * x.log(), -1)
def JSD(x): return H(x.mean(0)) - H(x).mean(0)


def one_hot_embedding(labels, nlabels):
    eye = torch.eye(nlabels)
    return eye[labels]


def max_comp(E, d):
    E = list(E)

    if len(E) == 0:
        for i in d.keys():
            E.append((i, i))
        return E, d

    graph = csr_matrix((np.ones(len(E)), zip(*E)), [np.max(E) + 1] * 2)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    max_elms = np.argwhere(labels == unique[np.argmax(counts)])

    max_ed_list = [e for e in E if (e[0] in max_elms) and (e[1] in max_elms)]

    dnew = dict([((int(k), d[k])) for k in max_elms.flatten()])

    return max_ed_list, dnew


class GKernel(nn.Module):
    def __init__(self, nodes, labels, filters=8, max_cc=None, hops=3, kernels='wl', normalize=True, store_fit=False, egonets_data=None, args={}):
        super(GKernel, self).__init__()
        self.hops = hops

        A = torch.from_numpy(np.random.rand(filters, nodes, nodes)).float()
        A = ((A + A.transpose(-2, -1)) > 1).float()
        A = torch.stack([a - torch.diag(torch.diag(a)) for a in A], 0)
        self.P = nn.Parameter(A, requires_grad=False)
        self.Padd = nn.Parameter(torch.randn(filters, nodes, nodes))
        self.Padd.data = (self.Padd.data + self.Padd.data.transpose(1, 2)) * 1e-2

        self.X = nn.Parameter(torch.stack([one_hot_embedding(torch.randint(labels, (nodes,)), labels)
                              for fi in range(filters)], 0), requires_grad=False)
        self.Xp = nn.Parameter(torch.zeros((filters, nodes, labels)).float(), requires_grad=True)

        self.temp = nn.Parameter(torch.tensor(10.0), requires_grad=True)

        self.filters = filters
        self.store = [None] * filters

        self.gks = []
        for kernel in kernels.split('+'):
            if kernel == 'wl':
                self.gks.append(lambda x: WeisfeilerLehman(n_iter=3, normalize=normalize))
            if kernel == 'wloa':
                self.gks.append(lambda x: WeisfeilerLehmanOptimalAssignment(n_iter=3, normalize=normalize))
            if kernel == 'prop':
                self.gks.append(lambda x: Propagation(normalize=normalize))
            if kernel == 'rw':
                self.gks.append(lambda x: RandomWalkLabeled(normalize=normalize))
            if kernel == 'gl':
                self.gks.append(lambda x: GraphletSampling(normalize=normalize, k=args.get('gl_k') or 5))
            if kernel == 'py':
                self.gks.append(lambda x: PyramidMatch(normalize=normalize))

        self.store_fit = store_fit
        self.stored = False

    def forward(self, x, edge_index, batch, not_used=None, fixedges=None, node_indexes=[], egonets_data=None, optimize=True):

        newA = (self.Padd > 0).float() * (1 - torch.eye(self.Padd.shape[-1])[None, ...])  # remove self loops
        self.P.data = newA
        self.X.data = (self.Xp * 1e6).softmax(-1)
        convs = []
        for gk in self.gks:
            convs.append(GKernelConv.apply(x, edge_index, batch, self.P, self.Padd, self.X, self.Xp,
                         self.hops, self.training, gk(None), self.stored, node_indexes, egonets_data, self.temp))
        conv = torch.cat(convs, -1)

        if not optimize:
            conv = conv.detach()

        return conv


def get_egonets(x, edge_index, i, hops=3):
    fn, fe, _, _ = torch_geometric.utils.k_hop_subgraph([i], num_hops=hops, edge_index=edge_index, num_nodes=x.shape[0])
    node_map = torch.arange(fn.max() + 1)
    node_map[fn] = torch.arange(fn.shape[0])
    ego_edges = node_map[fe]
    ego_nodes = x[fn, :]
    return ego_nodes, ego_edges


class GKernelConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index, batch, P, Padd, X, Xp, hops, training, gk, stored, node_indexes, egonets_data, temp):
        # graph similarity here
        filters = P.shape[0]
        convs = []

        if not stored:
            if egonets_data is None:
                egonets = [get_egonets(x, edge_index, i, hops) for i in torch.arange(x.shape[0])]
                def G1(i): return [set([(e[0], e[1]) for e in egonets[i][1].t().numpy()]),
                                   dict(zip(range(egonets[i][0].shape[0]), egonets[i][0].argmax(-1).numpy()))]
                Gs1 = [G1(i) for i in range(x.shape[0])]
            else:
                xcat = x.argmax(-1).numpy()
                Gs1 = [[set([(e[0], e[1]) for e in np.asarray(edg).T]), dict(
                    enumerate(xcat[nidx].reshape(len(nidx),)))] for nidx, edg in zip(*egonets_data)]

            conv = GKernelConv.eval_kernel(x, Gs1, P, X, gk, False)
        else:
            assert ('Should not happen')
            conv = GKernelConv.eval_kernel(None, None, P, X, gk, True)[node_indexes, :]
            Gs1 = None

        ctx.save_for_backward(x, edge_index, P, Padd, X, Xp, conv, batch)
        ctx.stored = stored
        ctx.node_indexes = node_indexes
        ctx.egonets_data = egonets_data
        ctx.Gs1 = Gs1
        ctx.gk = gk
        ctx.temp = temp

        return conv.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, edge_index, P, Padd, X, Xp, conv, batch = ctx.saved_tensors

        # grad_input -> kernel response gradient size: filters x nodes
        # perform random edit for each non zero filter gradient:
        grad_padd = 0
        grad_xp = 0

        kindexes = torch.nonzero(torch.norm(grad_output, dim=0))[:, 0]  # masks with non zero back-gradient

        ################### derivative wrt mask #############################
        edits = []
        edit_probs = []
        edit_vals = []
        for i in range(1):
            Pnew = P.detach().clone()
            Xnew = X.detach().clone()

            for fi in kindexes:
                edit_graph = torch.rand((1,)).item() < 0.5 or X.shape[-1] == 1
                Pnew, Xnew, _ = GKernelConv.random_edit(fi, Pnew, Padd, Xnew, Xp, edit_graph, 1, temp=1)
            if not ctx.stored:
                convnew = GKernelConv.eval_kernel(x, ctx.Gs1, Pnew, Xnew, ctx.gk, True)
            else:
                convnew = GKernelConv.eval_kernel(None, None, Pnew, Xnew, ctx.gk, True)[ctx.node_indexes, :]
            edit_prob = ((P - Pnew).abs() * Padd.sigmoid()).sum([-1, -2]) + \
                ((X - Xnew).abs() * Xp.softmax(-1)).sum([-1, -2])
            edits.append((Pnew, Xnew))
            edit_probs.append(torch.minimum(edit_prob, 1 - edit_prob) / torch.maximum(edit_prob, 1 - edit_prob))
            edit_vals.append(convnew)

        edit_probs = torch.stack(edit_probs, 0)[:, None, :]
        edit_vals = torch.stack(edit_vals, 0)
        I = torch.eye(edit_probs.shape[0])[:, :, None, None]

        for i, (Pnew, Xnew) in enumerate(edits):

            grad_fi = (conv + ((1 - I[i]) * edit_vals * edit_probs).sum(0)) / \
                (1 + ((1 - I[i]) * edit_probs).sum(0)) - (I[i] * edit_vals).sum(0)

            proj = (grad_fi * grad_output).sum(0)[:, None, None]

            grad_padd += proj * (P - Pnew)
            grad_xp += proj * (X - Xnew)

        ################### derivative wrt input featue #############################
        grad_inxp = 0
        for it in range(3):
            _, xpnew, _ = GKernelConv.random_edit(-1, None, None, x, x, False, 1, temp=0.1, batch=batch)

            xcat = xpnew.argmax(-1).numpy()
            Gs1 = [[set([(e[0], e[1]) for e in np.asarray(edg).T]), dict(enumerate(xcat[nidx].reshape(len(nidx),)))]
                   for nidx, edg in zip(*ctx.egonets_data)]
            convnew = GKernelConv.eval_kernel(xcat, Gs1, P, X, ctx.gk, False)

            grad_fi = conv - convnew
            proj = (grad_fi * grad_output).sum(0)[:, None, None]
            proj = proj * (x - xpnew)
            grad_inxp += proj * ((x).sigmoid() * (1 - (x).sigmoid()))
        ####################

        return grad_inxp, None, None, None, grad_padd * ((Padd).sigmoid() * (1 - (Padd).sigmoid())), None,\
            grad_xp * (Xp.sigmoid() * (1 - Xp.sigmoid())), None, None, None, None, None, None, None

    @staticmethod
    def eval_kernel(x, Gs1, P, X, gk, stored=False):
        filters = P.shape[0]
        nodes = P.shape[1]

        Gs2 = [max_comp(set([(e[0], e[1]) for e in torch_geometric.utils.dense_to_sparse(P[fi])[0].t().numpy()]),
                        dict(zip(range(nodes), X[fi].argmax(-1).flatten().detach().numpy()))) for fi in range(filters)]

        if not stored:
            gk.fit(Gs1)
            try:
                sim = gk.transform(Gs2)
            except:
                print('---------------------------------------------------------')
                print(Gs1)
                print('---------------------------------------------------------')
                print(Gs2)
                print('---------------------------------------------------------')
                input('stopping here for inpsection')
            sim = np.nan_to_num(sim)
        else:
            sim = gk.transform(Gs2)
            sim = np.nan_to_num(sim)

        return torch.from_numpy(sim.T)

    @staticmethod
    def random_edit(i, Pin, Padd, X, Xp, edit_graph, n_edits=1, temp=1, batch=None):
        if i == -1:
            X = X.clone()
            PX = (Xp.double() * temp).softmax(-1).data

            for bi in range(batch.max() + 1):
                bM = np.nonzero(batch == bi).flatten()

                pi = 1 - PX[bM].max(-1)[0] + 1e-8
                pi = pi / (pi.sum(-1, keepdims=True))

                lab_ind = np.random.choice(pi.shape[0], (n_edits,), p=pi)
                lab_val = [np.random.choice(PX.shape[-1], size=(1,), replace=False, p=PX[bM[j]]) for j in lab_ind]

                X.data[bM[lab_ind], :] = 0
                X.data[bM[lab_ind], lab_val] = 1

            return Pin, X, None

        P = Pin

        if edit_graph:  # edit graph
            P = Pin.clone()
            Pmat = P[i] * (1 - (Padd[i] * temp).sigmoid()).data + (1 - P[i]) * \
                (Padd[i] * temp).sigmoid().data  # sample edits
            Pmat = Pmat * (1 - np.eye(Pmat.shape[-1])) + 1e-20

            PmatN = Pmat / Pmat.sum()
            inds = np.random.choice(Pmat.shape[0]**2, size=(n_edits,), replace=False, p=PmatN.flatten().numpy(),)
            edit_prob = Pmat.flatten().numpy()[inds]
            inds = torch.from_numpy(np.stack(np.unravel_index(inds, Pmat.shape), 0)).to(Pmat.device)

            inds = torch.cat([inds, inds[[1, 0], :]], -1)  # symmetric edit
            P[i].data[inds[0], inds[1]] = 1 - P[i].data[inds[0], inds[1]]

            if (P[i].sum() == 0):  # avoid fully disconnected graphs
                P = Pin.clone()

        else:  # edit labels
            X = X.clone()
            PX = (Xp[i] * temp).softmax(-1).data

            pi = 1 - PX.max(-1)[0] + 1e-8
            pi = pi / (pi.sum(-1, keepdims=True))

            PXnew = PX * (1 - X[i]) + 1e-8
            PXnew = PXnew / PXnew.sum(-1, keepdims=True)

            lab_ind = np.random.choice(X[i].shape[0], (n_edits,), p=pi.numpy())
            lab_val = [np.random.choice(PX.shape[1], size=(1,), replace=False, p=PXnew[j, :].numpy(),) for j in lab_ind]

            X[i].data[lab_ind, :] = 0
            X[i].data[lab_ind, lab_val] = 1

            edit_prob = np.asarray([PX[li, lv].numpy() for li, lv in zip(lab_ind, lab_val)])

        return P, X, edit_prob


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        print(hparams)

        try:
            in_features, hidden, num_classes, labels = hparams.in_features, hparams.hidden, hparams.num_classes, hparams.labels
        except:
            hparams = Namespace(**hparams)
            in_features, hidden, num_classes, labels = hparams.in_features, hparams.hidden, hparams.num_classes, hparams.labels

        self.save_hyperparameters(hparams)

        self.conv_layers = nn.ModuleList()
        self.vq_layers = nn.ModuleList()

        self.conv_layers.append(GKernel(hparams.nodes, labels, hidden, max_cc=self.hparams.max_cc,
                                hops=hparams.hops, kernels=hparams.kernel, store_fit=True, args=vars(hparams)))

        n_kernels = len(hparams.kernel.split('+'))
        for i in range(1, hparams.layers):
            self.conv_layers.append(GKernel(hparams.nodes, hidden, hidden, max_cc=self.hparams.max_cc,
                                    hops=hparams.hops, kernels=hparams.kernel, args=vars(hparams)))

            self.vq_layers.append(SoftAss(hidden, hidden))

        activation = nn.LeakyReLU
        if hparams.activation == 'sigmoid':
            activation = nn.Sigmoid

        if hparams.mlp_layers == 2:
            self.fc = nn.Sequential(nn.Linear(hidden * n_kernels * hparams.layers, hidden),
                                    activation(), nn.Linear(hidden, hidden), activation(), nn.Linear(hidden, num_classes))
        else:
            self.fc = nn.Sequential(nn.Linear(hidden * n_kernels, hidden), activation(), nn.Linear(hidden, num_classes))

        self.eye = torch.eye(hidden)
        self.lin = nn.Linear(hidden, hidden)

        self.automatic_optimization = False

        def _regularizers(x):
            jsdiv = hparams.jsd_weight * JSD(x.softmax(-1) + 1e-20)
            return -jsdiv  # + maxresp
        self.regularizers = _regularizers

        self.mask = nn.Parameter(torch.ones(hidden).float())

    def prefit(self, data):
        for l in self.conv_layers:
            if l.store_fit:
                with torch.no_grad():
                    l.prefit(data.x, data.edge_index)

    def one_hot_embedding(self, labels):
        self.eye = self.eye.to(labels.device)
        return self.eye[labels]

    def forward(self, data):

        if 'nidx' not in data.__dict__:
            data.nidx = None

        batch = data.batch
        edge_index = data.edge_index
        x = data.x

        egonets_data = [data.egonets_idx, data.egonets_edg]

        loss = x.sum().detach() * 0

        responses = []
        for l, vq in zip(self.conv_layers, [None] + list(self.vq_layers)):  # only works with one layer
            x = l(x, edge_index, node_indexes=data.nidx, egonets_data=egonets_data,
                  batch=batch, optimize=self.hparams.optimize_masks)

            if self.mask is not None:
                x = x * self.mask[None, :].repeat(1, x.shape[-1] // self.mask.shape[-1])

            responses.append(x)
        x = torch.cat(responses, -1)

        pooling_op = None
        if self.hparams.pooling == 'add':
            pooling_op = global_add_pool
        if self.hparams.pooling == 'max':
            pooling_op = global_max_pool
        if self.hparams.pooling == 'mean':
            pooling_op = global_mean_pool

        return self.fc(pooling_op(x, batch)), responses, loss

    def configure_optimizers(self):
        graph_params = set(self.conv_layers.parameters())
        cla_params = set(self.parameters()) - graph_params
        optimizer = torch.optim.Adam([{'params': list(graph_params), 'lr': self.hparams.lr_graph},
                                      {'params': list(cla_params), 'lr': self.hparams.lr}])

        return optimizer

    def training_step(self, train_batch, batch_idx):

        data = train_batch

        optimizer = self.optimizers()

        optimizer.zero_grad()

        output, responses, vqloss = self(data)
        loss_ce = self.hparams.loss(torch.squeeze(output, -1), data.y)

        loss_jsd = torch.stack([self.regularizers(x) for x in responses]).mean()

        loss = loss_ce + loss_jsd  # + vqloss +  loss_spa
        loss.backward()
        optimizer.step()

        acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
        self.log('acc', acc, on_step=False, on_epoch=True)
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('loss_jsd', loss_jsd.item(), on_step=False, on_epoch=True)
        self.log('loss_ce', loss_ce.item(), on_step=False, on_epoch=True)

    def validation_step(self, train_batch, batch_idx):
        data = train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.hparams.loss(torch.squeeze(output, -1), data.y)
            acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
            self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True)

    def test_step(self, train_batch, batch_idx):
        data = train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.hparams.loss(torch.squeeze(output, -1), data.y)
            acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
            self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)


class SoftAss(nn.Module):
    def __init__(self, num_words, features_dim, softmax=True):
        super(SoftAss, self).__init__()

        if softmax:
            self.normalize = nn.Softmax(-1)
        else:
            self.normalize = nn.Normalize(dim=-1)

        self.dict = nn.Parameter(torch.rand(num_words, features_dim).float(), requires_grad=True)
        self.codebook_init = False

    def reset_codebook(self, x):
        if self.codebook_init:
            centroid, label = kmeans2(x.detach().cpu().numpy(), self.dict.detach().cpu().numpy(), minit='matrix')
        else:
            centroid, label = kmeans2((x + torch.randn_like(x) * 1e-4).detach().cpu().numpy(), self.dict.shape[0])
        self.dict.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True

    def forward(self, x):
        if self.training:
            self.reset_codebook(x)

        _x = self.normalize(x)

        res = torch.tensordot(_x, self.dict, dims=([1], [1]))

        return res
