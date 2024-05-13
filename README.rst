==========
README
==========


Neurograph is a library for training and evaluating Graph Neural Network,
Transformer and (optionally) CW-network models on fMRI and DTI data.

Install to virtual env
----------------------

``neurograph`` requires: python=3.10, torch=1.12.1, cuda>=11.3

GNN and Transformers only
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # install Pytorch Geometric
   ./install_pyg.sh cu113  # or ./install_pyg.sh cpu

   # install other requirements
   pip install -r requirements.txt

   # install `neurograph` into env
   pip install -e .

(Optional) CWN
~~~~~~~~~~~~~~

In order to install CWN, you need to install extra dependencies, namely
``conda``, ``graph-tool`` and the fork of ``cwn`` library.

First, install conda with ``python=3.10`` as decribed here
`Conda installation <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
Then create conda env and run commands below.

.. code:: bash

   ./graph-tool_install.sh

   # install Pytorch Geometric
   ./install_pyg.sh cu113  # or ./install_pyg.sh cpu

   # install other requirements
   pip install -r requirements.txt

   # install `neurograph` into env
   pip install -e .

   # CWN: install the fork of `cwn` library
   cd ..
   git clone https://github.com/gennadylaptev/cwn.git
   cd cwn
   pip install -e .

   # if you encouter any errors with conda glibcxxx headers
   # reinstall scipy
   pip install -I scipy==1.10.1


Docker
------

.. code:: bash

   cd to the root directory of neurograph repo

   # build an image
   docker build -t neurograph .

   # run it as a container and do your stuff inside
   docker run -it --rm --network host --gpus=all -v $(pwd):/app neurograph /bin/bash

   # or run a particular gridsearch e.g.
   docker run --rm --network host --gpus=0,1 -v $(pwd):/app --env WANDB_API_KEY=<YOUR_WANDB_API_KEY> neurograph bash -c 'python -m neurograph.train --multirun log.wandb_project=mri_docker_test +model=transformer8,transformer16 ++model.num_layers=1 model.num_heads=1,8 dataset.data_type=dense train.scheduler=null'


Datasets
--------
Currently, library supports 4 datasets: COBRE, PPMI, ABIDE, HCP.

By default, neurograph expects that your datasets are stored in
``datasets`` folder e.g. ``{NEUROGRAPH_ROOT}/datasets/cobre``.

``neurograph`` expects a particular directory structure for a dataset:

::

   datasets/{DATASET_NAME}
   ├── {DATASET_NAME}_splits.json  # JSON file w/ cross-validation splits
   ├── dti  # experiment name
   │   ├── processed
   │   └── raw  # atlas name; this dir contains connectivity matrices: a separate file for each subject
   ├── fmri  # experiment name
   │   ├── processed
   │   └── raw
   │       └── aal  # atlas name; this dir contains connectivity matrices: a separate file for each subject
   └── meta_data.tsv  # file with targets and other meta info


Models
------

``neurograph`` supports the following models for unimodal data: ``GCN``, ``GAT``
(from `pytorch geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_),
``GCN`` and ``GAT`` from `BrainGB <https://arxiv.org/abs/2204.07054>`_
and standart Transformer (based on `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
and `On Layer Normalization in the Transformer Architecture <https://arxiv.org/pdf/2002.04745.pdf>`_).
Optionally, you can also ``Cellular Invariant Model (CIN)`` from
`Weisfeiler and Lehman Go Cellular: CW Networks <https://openreview.net/forum?id=uVPZCMVtsSG>`_.

For multimodal data, `neurograph` only supports Transformer with several different mechanisms
(cross-attention and hierarchical attention) for fusion of modalities.


Usage
-----

Neurograph uses `hydra <https://hydra.cc/>`_  for managing different configurations. See
default config in ``config/config.py`` and ``config/config.yaml``.
By default, all results (all config parameters and final metrics)
are logged to ``wandb`` and stored locally to ``multirun`` dir.

Settings for ``wandb`` are set in ``config.log``` section.

``hydra`` will create a separate dir for each experiment with the following name
``multirun/${now:%Y-%m-%d}/{now:%H-%M-%S}/{num of experiment}``.
For each experiment three files will be saved:
final metrics on validation and tests sets will in ``metrics.json``,
copy of config as ``config.yaml`` and stdout as ``__main__.log``.

To run a particular gridsearch (GNN or Transformers), you must define dataset type and name as well as model name:

.. code:: bash

   python -m neurograph.train --multirun +train=base_train +model=MODEL_NAME +dataset=DATASET_TYPE dataset.name=DATASET_NAME dataset.data_type=DATA_TYPE ...

where ``MODEL_NAME={standartGNN,bgbGAT,bgbGCN,transformer}``,
``DATASET_TYPE={base_dataset, base_multimodal_dataset, cellular_dataset}``,
``DATASET_NAME={abide, cobre, ppmi}``
``DATA_TYPE={graph, dense}``.

Gridsearch for CIN model is launched by a different command (see below for an example).

Below are examples of commands to run training for a particular dataset
in unimodal or multimodal setting.

Unimodal experiments
~~~~~~~~~~~~~~~~~~~~
COBRE dataset is used by default.

For example, in order to run gridsearch for standart GAT or GCN:

.. code:: bash

   python -m neurograph.train --multirun +train=base_train +model=standart_gnn model.layer_module=GATConv model.num_layers=1,2 model.num_heads=1,2,4 model.hidden_dim=8,12,16 +dataset=base_dataset dataset.pt_thr=0.25,0.5,0.75,null train.epochs=20 train.scheduler=null

Set ``model.layer_module=GCNConv`` for GCN or ``model.layer_module=GATConv`` for GAT

For BrainGB GCN, BrainGB GAT use:

.. code:: bash

   python -m neurograph.train --multirun +train=base_train +model=bgbGAT model.num_layers=1,2 model.num_heads=1,2,4 model.hidden_dim=8,12,16 +dataset=base_dataset dataset.pt_thr=0.25,0.5,0.75,null train.epochs=20 train.scheduler=null

Set ``+model=bgbGAT`` for ``BrainGB GAT`` or ``+model=bgbGCN`` for ``BrainGB GCN``.

For vanilla transformers, run

.. code:: bash

   python -m neurograph.train --multirun +train=base_train dataset.data_type=dense +model=transformer8,transformer16,transformer32,transformer64,transformer116 model.num_layers=1,2 model.num_heads=1,2,4 model.pooling=concat,mean +dataset=base_dataset dataset.feature_type=conn_profile,timeseries train.scheduler=null train.device="cuda:0" train.epochs=100


Multimodal experiments
~~~~~~~~~~~~~~~~~~~~~~

To run multimodal experiments you need to specify a different base
dataset config: ``+dataset=base_multimodal_dataset`` and use a
corresponding multimodal model (e.g. ``+model=mm_transformer``; see ``neurograph.models``)


Build documentation
-------------------

.. code:: bash

   cd docs
   make html

Documentation will be stored to ``docs/build``


Acronyms used throughout the code
---------------------------------

-  PyG = pytorch_geometric

-  CM = connectivity matrix

-  MP = message passing

-  MM = multimodal (e.g. MM2 - multimodal, 2 modalities)

-  subset = train, valid or test part of a whole dataset or of one fold
   in cross-validation
