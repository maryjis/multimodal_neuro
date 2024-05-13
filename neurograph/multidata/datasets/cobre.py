""" COBRE dataset classes """

from neurograph.unidata.datasets.cobre import CobreTrait
from neurograph.multidata.dense import MutlimodalDense2Dataset, MutlimodalDenseMorphDataset


class CobreMultimodalDense2Dataset(CobreTrait, MutlimodalDense2Dataset):
    """Multimodal dense dataset w/ 2 modalities (fmri+dti) for COBRE dataset"""

class CobreMultimodalMorphDense2Dataset(CobreTrait, MutlimodalDenseMorphDataset):
    """Multimodal dense dataset w/ 2 modalities (fmri+morph)  for COBRE dataset"""