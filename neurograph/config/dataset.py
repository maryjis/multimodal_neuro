"""Dataclasses describing configs for different types of datasets"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import neurograph

DEFAULT_DATA_PATH = Path(neurograph.__file__).resolve().parent.parent / "datasets"


@dataclass
class DatasetConfig:
    """Base dataset config class

    Attributes:
        data_type (str): type of data `graph` or `dense
        name (str): dataset name (default `cobre`)
        atlas (str): atlas name (default `aal`)
        normalize (str, optional): normalization method for DTI experiment
            (default None, options={log, global_max})
        data_path (Path): path to datasets folder (default `root/datasets`)
    """

    data_type: str

    name: str = "cobre"
    atlas: str = "aal"

    # for DTI
    normalize: Optional[str] = None

    data_path: Path = DEFAULT_DATA_PATH


@dataclass
class UnimodalDatasetConfig(DatasetConfig):
    """Dataset config for Unimodal data both graph
    (pytorch_geometric.Data object) and dense (just torch.Tensor objects)

    Attributes:
        data_type (str): type of data (default `graph`, options={'graph', 'dense'})
        experiment_type (str): {'fmri', 'dti'}
        abs_thr (float, optional): absolute threshold for pruning edges,
            only for graph data (default None)
        pt_thr (float, optional): proportional threshold for pruning edges,
            only for graph data (default None)
        feature_type: str = feature type for tokens,
            only for dense data (default "conn_profile", options={'conn_profile', 'timeseries'}
    """

    data_type: str = "graph"  # or 'dense'
    experiment_type: str = "fmri"
    # graph specific
    # init_node_features: str = 'conn_profile'
    # graph specific
    abs_thr: Optional[float] = None
    pt_thr: Optional[float] = None
    # dense specific
    feature_type: str = "conn_profile"  # 'timeseries'
    # augmentation parameters
    time_series_length: Optional[int] = None
    random_crop: bool = False
    random_crop_strategy: str = "uniform"  # "per_roi"
    # post processing
    scale: bool = False


@dataclass
class TimeSeriesDenseDatasetConfig(DatasetConfig):
    """Config for creating DenseTimeSeriesDataset instance"""

    data_type: str = "DenseTimeSeriesDataset"
    experiment_type: str = "fmri"
    fourier_timeseries: bool = True


@dataclass
class MultiGraphDatasetConfig(DatasetConfig):
    """Config for creating MultiGraphDataset instance"""

    data_type: str = "MultiGraphDataset"

    abs_thr: Optional[float] = None
    pt_thr: Optional[float] = None
    normalize: str = "global_max"
    fusion: str = "dti_binary_mask"


@dataclass
class MultimodalDatasetConfig(DatasetConfig):
    """Dataset config for Dense Multimodal data (fMRI and DTI)

    Attributes:
        data_type (str): type of data (default `graph`, options={'graph', 'dense'})
        fmri_feature_type: str = feature type for tokens,
            only for dense data (default "conn_profile", options={'conn_profile', 'timeseries'}
        normalize (str, optional): normalization method for DTI experiment
            (default None, options={log, global_max})
    """

    data_type: str = "multimodal_dense_2"  # only option rn
    fmri_feature_type: str = "conn_profile"
    normalize: Optional[str] = "global_max"
    time_series_length: Optional[int] = None
    
@dataclass
class MultimodalMorphDatasetConfig(DatasetConfig):
    """Dataset config for Dense Multimodal data (fMRI and T1 morphometry)

    Attributes:
        data_type (str): type of data (default `graph`, options={'dense'})
        fmri_feature_type: str = feature type for tokens,
            only for dense data (default "conn_profile", options={'timeseries'}
        normalize (str, optional): normalization method for DTI experiment
            (default None, options={log, global_max})
    """

    data_type: str = "morph_multimodal_dense_2"  # only option rn
    fmri_feature_type: str = "timeseries"
    normalize: Optional[str] = "global_max"
    time_series_length: Optional[int] = None        


@dataclass
class CellularDatasetConfig:
    """Dataset config for cell complexes dataset

    Attributes:
        name (str): dataset name (default `cobre`)
        pt_thr (float, optional): proportional threshold for pruning edges, (default None)
        max_ring_size (int): max ring size for building 2-cells in a complex (default 3)
        n_jobs (int): num of jobs used for building 2-complexes out of graphs (default 1)
        top_pt_rings (float): Proportional threshold for rings (default 0.01)
        data_path (Path): path to datasets folder (default `root/datasets`)
    """

    name: str = "cobre"  # ignored
    pt_thr: Optional[float] = None
    max_ring_size: int = 3
    n_jobs: int = 1
    top_pt_rings: float = 0.1

    data_path: Path = DEFAULT_DATA_PATH
