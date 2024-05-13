# we need to import graph_tool first, otherwise we'll get an error
try:
    import graph_tool as gt
except Exception as err:
    pass

import torch
from torch_geometric import seed_everything

seed_everything(1380)
torch.set_num_threads(1)
