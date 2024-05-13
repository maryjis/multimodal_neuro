""" Available dict w/ pointers to available Pytorch and Pytorch Geometric classes """

import inspect
from typing import Any, Type
from torch import nn, optim
from torch.optim import lr_scheduler
from torch_geometric import nn as tg_nn
from neurograph.loss import custom_loss
from neurograph.models.snake import Snake

# put different pytorch models into dicts for easier instantiation
available_pg_modules: dict[str, Type[nn.Module]] = {
    name: obj
    for (name, obj) in inspect.getmembers(tg_nn)
    if inspect.isclass(obj)
    if issubclass(obj, nn.Module)
}

available_modules: dict[str, Type[nn.Module]] = {
    name: obj
    for (name, obj) in inspect.getmembers(nn)
    if inspect.isclass(obj)
    if issubclass(obj, nn.Module)
}

available_activations: dict[str, Type[nn.Module]] = {
    **{
        name: obj
        for (name, obj) in available_modules.items()
        if obj.__module__.endswith("activation")
    },
    **{"Snake": Snake},
}

available_losses: dict[str, Type[nn.Module]] = {
    name: obj for name, obj in available_modules.items() if name.endswith("Loss")
}
available_custom_losses: dict[str, Type[nn.Module]] = {
    name: obj
    for (name, obj) in inspect.getmembers(custom_loss)
    if inspect.isclass(obj)
    if issubclass(obj, nn.Module)
}
available_losses: dict[str, Type[nn.Module]] = {
    **available_losses,
    **available_custom_losses,
}

available_optimizers: dict[str, Type[optim.Optimizer]] = {
    name: obj
    for (name, obj) in inspect.getmembers(optim)
    if inspect.isclass(obj)
    if issubclass(obj, optim.Optimizer)
}

# there's no base class for all schedulers
available_schedulers: dict[str, Any] = {
    name: obj
    for (name, obj) in inspect.getmembers(lr_scheduler)
    if inspect.isclass(obj)
}
