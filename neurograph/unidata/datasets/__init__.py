""" Module that contains implementations for each dataset """

import inspect
from typing import Type

from neurograph.unidata.base_dataset import BaseDataset
from neurograph.unidata.graph import UniGraphDataset
from neurograph.unidata.datasets import abide, cobre, ppmi, hcp, alexeev
from neurograph.unidata.dense import UniDenseDataset, DenseTimeSeriesDataset

dataset_list = [cobre, abide, ppmi, hcp, alexeev]

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
dense_datasets: dict[str, Type[UniDenseDataset]] = {
    ds_name: obj
    for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == "dense"
    if issubclass(obj, UniDenseDataset)
}
dense_ts_datasets: dict[str, Type[DenseTimeSeriesDataset]] = {
    ds_name: obj
    for (ds_name, data_type), obj in available_datasets.items()
    if issubclass(obj, DenseTimeSeriesDataset)
}
graph_datasets: dict[str, Type[UniGraphDataset]] = {
    ds_name: obj
    for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == "graph"
    if issubclass(obj, UniGraphDataset)
}
traits: dict = {
    obj.name: obj
    for modules in dataset_list
    for (class_name, obj) in inspect.getmembers(modules)
    if inspect.isclass(obj)
    if class_name.endswith("Trait")
}
