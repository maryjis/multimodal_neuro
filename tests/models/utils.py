import numpy as np
import torch
from torch_geometric.data import Data, Batch
from neurograph.unidata.utils import conn_matrix_to_edges


def random_graph(n, f=13):
    # generate weights in range [-1, 1]
    w = 2 * np.random.rand(n, n) - 1.0
    edge_index, edge_attr = conn_matrix_to_edges(w)

    return Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=torch.randn(n, f).float(),
        num_nodes=n,
        y=torch.randint(0, 2, (n,)),
        subj_id=[str(i) for i in range(n)],
    )


def random_pyg_batch(batch_size=3, num_nodes=17, num_features=13):
    return Batch.from_data_list(
        [random_graph(num_nodes, num_features) for _ in range(batch_size)],
    )
