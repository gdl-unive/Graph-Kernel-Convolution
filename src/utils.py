from torch_scatter import scatter

### useful functions from torch_geometric.nn ###


def global_add_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False,
             num_nodes=None):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes
