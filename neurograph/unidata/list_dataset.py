"""Simple pytorch geometric ListDataset (dataset initialized from a list of Data objects)
used for quick experiments
"""

import torch
from torch_geometric.data import Data, InMemoryDataset


# We don't use abstract methods `download` and `raw_file_names`
# so we do not override them.
# pylint: disable=abstract-method
class ListDataset(InMemoryDataset):  # pragma: no cover
    """Basic dataset for ad-hoc experiments"""

    def __init__(self, root, data_list: list[Data]):
        # first store `data_list` as attr
        self.data_list = data_list
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])
