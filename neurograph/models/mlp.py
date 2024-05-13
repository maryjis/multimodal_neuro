""" Module offers basic MLP class that can be easily created from config """

import torch
from torch import nn

from neurograph.config import MLPConfig, MLPlayer
from neurograph.models.available_modules import available_activations


def build_mlp_layer(in_size: int, layer: MLPlayer) -> nn.Sequential:
    """Factory function that returns nn.Sequential from input size and MLPlayer"""
    act_params = layer.act_func_params if layer.act_func_params else {}

    lst: list[nn.Module] = [nn.Linear(in_size, layer.out_size)]
    # lst.append(
    if layer.act_func:
        lst.append(available_activations[layer.act_func](**act_params))
    if layer.dropout:
        lst.append(nn.Dropout(layer.dropout, inplace=True))
    return nn.Sequential(*lst)


class BasicMLP(nn.Module):
    """Basic MLP class"""

    def __init__(self, in_size: int, out_size: int, config: MLPConfig):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.net = nn.Sequential()
        curr_size = self.in_size
        for layer_conf in config.layers:
            subnet = build_mlp_layer(curr_size, layer_conf)
            self.net.append(subnet)
            curr_size = layer_conf.out_size

        # the last layer
        self.net.append(
            build_mlp_layer(
                curr_size,
                MLPlayer(
                    out_size=self.out_size,
                    act_func=config.act_func,
                    act_func_params=config.act_func_params,
                ),
            )
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# IT DOESNT WORK NOW
# def mirror_mlp_config(conf: MLPConfig) -> MLPConfig:
#    sizes = []
#    for l in reversed(conf.layers):
#        sizes.append(l.out_size)
#    sizes.append(self.in_size)
#
#    new_in_size = sizes[0]
#    rlayers = []
#    for rsize, layer_cfg in zip(sizes[1:], self.layers):
#        new_cfg = deepcopy(layer_cfg)
#        new_cfg.out_size = rsize
#        rlayers.append(new_cfg)
#
#    return MLPConfig(new_in_size, rlayers)
