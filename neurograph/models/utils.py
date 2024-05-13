""" Util functions used in models code """

import torch


def concat_pool(x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Function to apply concat pooling to GNN output embeddings"""

    # NB: x must be a batch of xs
    return x.reshape(x.size(0) // num_nodes, -1)
