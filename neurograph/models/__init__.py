""" Module w/ all model classes
Please, import all the model class into __init__
"""

from .gat import BrainGAT
from .gcn import BrainGCN
from .mlp import BasicMLP
from .transformers import Transformer
from .multimodal_transformers import MultiModalTransformer
from .standart_gnn import StandartGNN
from .dummy import DummyMultimodalDense2Model
from .prior_transformers import PriorTransformer
from .bolT import BolT
from .temporal_cnn import TemporalCNN
from .conformer import Conformer
from .multiBolt import MultiModalBolt, MultiModalMorphBolt, MultiModalMorphBoltV2
from .fedformer.fedformer import FEDformer
from .gtn import GTN, CustomGTN

# graph_model_classes = {
#    'bgbGAT': bgbGAT,
#    'bgbGCN': bgbGCN,
# }
# dense_model_classes = {
#    'MLP': BasicMLP,
#    'transformer': Transformer,
# }
