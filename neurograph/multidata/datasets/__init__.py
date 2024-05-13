""" Module that contains dataset classes """

import inspect
from typing import Type

from neurograph.multidata.datasets import cobre
from neurograph.unidata.datasets import (
    BaseDataset,
)
from neurograph.multidata.dense import MutlimodalDense2Dataset, MutlimodalDenseMorphDataset

dataset_list = [cobre]

available_datasets: dict[tuple[str, str], Type[BaseDataset]] = {
    (obj.name, obj.data_type): obj
    for modules in dataset_list
    for (class_name, obj) in inspect.getmembers(modules)
    if inspect.isclass(obj)
    if issubclass(obj, BaseDataset)
    if hasattr(obj, "name") and hasattr(obj, "data_type")
    if obj.name and obj.data_type
    if class_name.endswith("Dataset")
}
multimodal_dense_2: dict[str, Type[MutlimodalDense2Dataset]] = {
    ds_name: obj
    for (ds_name, data_type), obj in available_datasets.items()
    if issubclass(obj, MutlimodalDense2Dataset)
}

morph_multimodal_dense_2: dict[str, Type[MutlimodalDenseMorphDataset]] = {
    ds_name: obj
    for (ds_name, data_type), obj in available_datasets.items()
    if issubclass(obj, MutlimodalDenseMorphDataset)
}

