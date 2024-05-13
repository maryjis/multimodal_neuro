lint:
	pylint neurograph

test:
	pytest -v -s --disable-warnings --cov-report term-missing --cov=. tests/

test_train_bgbGCN:
	python -m neurograph.train log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +model=bgbGCN +dataset=base_dataset train.device=cpu

test_train_bgbGAT:
	python -m neurograph.train hydra.job_logging.root.level=DEBUG log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +model=bgbGAT +dataset=base_dataset train.device=cpu

test_train_standart_GAT:
	python -m neurograph.train log.wandb_mode=disabled check_commit=false train=base_train train.num_outer_folds=5 model=standart_gnn model.layer_module=GATConv model.num_heads=2 dataset=base_dataset train.device=cpu train.epochs=1

test_train_standart_GCN:
	python -m neurograph.train log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5  +model=standart_gnn model.layer_module=GCNConv +dataset=base_dataset train.device=cpu train.epochs=1

test_train_multi_bgbGAT:
	python -m neurograph.train log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +model=bgbGAT +dataset=multigraph_dataset train.device=cpu

test_train_transformer:
	python -m neurograph.train \
		log.wandb_mode=disabled check_commit=false \
		+train=base_train train.num_outer_folds=5 train.device=cpu\
		+model=transformer8 \
		+dataset=base_dataset dataset.data_type=dense \
		dataset.feature_type=timeseries \
		dataset.random_crop=true \
		dataset.time_series_length=100

test_train_prior_transformer:
	python -m neurograph.train --multirun log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +model=prior_transformer8,prior_transformer16 ++model.alpha=1 ++model.trainable_alpha=False +dataset=ts_dataset train.device=cpu

test_train_transformer_mm2:
	python -m neurograph.train --multirun log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +dataset=base_multimodal_dataset dataset.name=cobre +model=mm_transformer model.make_projection=True model.attn_type='hierarchical','cross-attention' model.num_layers=1 model.num_heads=2 model.pooling=concat model.hidden_dim=8 train.epochs=1 train.device=cpu

test_train_bolt:
	python -m neurograph.train check_commit=false log.wandb_mode=disabled +model=bolt +train=bolt_train +dataset=base_dataset dataset.data_type=dense dataset.feature_type=timeseries dataset.name=hcp dataset.atlas=shen  dataset.time_series_length=150 train.device=cpu dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets train.batch_size=8 train.epochs=1 model.nOfLayers=1 train.num_outer_folds=4 model.numHeads=4 model.headDim=2 model.windowSize=70

test_train_cwn:
	python -m neurograph.run_cwn log.wandb_mode=disabled check_commit=false +train=base_train train.num_outer_folds=5 +model=sparse_cin model.hidden=4 model.num_layers=1 +dataset=cellular_dataset dataset.pt_thr=0.05 dataset.top_pt_rings=0.1 dataset.n_jobs=1 dataset.max_ring_size=3 train.device=cpu train.epochs=1 train.num_threads=1 train.batch_size=32 train.valid_batch_size=32


test_train_gnn: test_train_bgbGCN test_train_bgbGAT test_train_multi_bgbGAT test_train_standart_GCN test_train_standart_GAT

test_train_transformers: test_train_transformer test_train_transformer_mm2 test_train_bolt

test_train: test_train_gnn test_train_transformers


test_abide_graph:
	python -m neurograph.train \
		check_commit=false \
		+model=bgbGAT model.num_layers=1 model.hidden_dim=8 model.num_heads=2 \
		+dataset=base_dataset dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets \
		dataset.data_type=graph dataset.name=abide dataset.atlas=schaefer \
		dataset.pt_thr=0.25 \
		+train=base_train train.batch_size=8 train.valid_batch_size=8 \
		train.scheduler=null train.num_threads=1 \
		train.optim_args.lr=1e-4 train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		log.wandb_mode=disabled train.epochs=1 train.device="cpu"

test_abide_dense_ts_no_scale:
	python -m neurograph.train \
		check_commit=false \
		+model=transformer8 \
		+dataset=base_dataset dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets \
		dataset.data_type=dense dataset.name=abide dataset.atlas=schaefer \
		dataset.time_series_length=78 dataset.feature_type=timeseries \
		+train=base_train train.batch_size=8 train.valid_batch_size=8 \
		train.scheduler=null train.num_threads=1 \
		train.optim_args.lr=1e-4 train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		log.wandb_mode=disabled train.epochs=1 train.device="cpu"

test_abide_dense_ts_scale:
	python -m neurograph.train \
		check_commit=false \
		+model=transformer8 \
		+dataset=base_dataset dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets \
		dataset.data_type=dense dataset.name=abide dataset.atlas=schaefer \
		dataset.time_series_length=78 dataset.feature_type=timeseries \
		dataset.scale=true \
		+train=base_train train.batch_size=8 train.valid_batch_size=8 \
		train.scheduler=null train.num_threads=1 \
		train.optim_args.lr=1e-4 train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		log.wandb_mode=disabled train.epochs=1 train.device="cpu"

test_abide_dense_cp:
	python -m neurograph.train \
		check_commit=false \
		+model=transformer8 \
		+dataset=base_dataset dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets \
		dataset.data_type=dense dataset.name=abide dataset.atlas=schaefer \
		dataset.feature_type=conn_profile \
		+train=base_train train.batch_size=8 train.valid_batch_size=8 \
		train.scheduler=null train.num_threads=1 \
		train.optim_args.lr=1e-4 train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		log.wandb_mode=disabled train.epochs=1 train.device="cpu"

temp_cnn:
	python -m neurograph.train \
		check_commit=false \
		log.wandb_mode=disabled \
		model=temp_cnn \
		dataset.data_type=dense dataset.name=cobre \
		dataset.feature_type=timeseries train.num_threads=1 \
		train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		train.epochs=20 train.device="cpu"

conformer:
	python -m neurograph.train \
		check_commit=false \
		log.wandb_mode=disabled \
		model=conformer \
		model.num_heads=2 \
		model.ffn_dim=50 \
		model.act_func_name=Snake \
		+'model.act_func_params={a: 5.0}' \
		dataset.data_type=dense dataset.name=cobre \
		dataset.feature_type=timeseries \
		train.device="cpu" train.num_threads=1 \
		train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=10 \
		train.epochs=1

fedformer:
	python -m neurograph.train \
		check_commit=false \
		log.wandb_mode=disabled \
		model=fedformer \
		model.hidden_dim=116 \
		model.num_heads=2 \
		model.num_modes=4 \
		model.attn_block_type=FEBw \
		model.L=3 \
		model.k=4 \
		model.c=16 \
		model.base=legendre \
		dataset.data_type=dense dataset.name=cobre \
		dataset.feature_type=timeseries \
		train.device="cpu" train.num_threads=1 \
		train.optim_args.weight_decay=0.01 \
		train.select_best_metric=loss train.num_outer_folds=4 \
		train.epochs=1

gtn:
	python -m neurograph.train \
		check_commit=false \
		log.wandb_mode=disabled \
		model=gtn \
		dataset.data_type=dense dataset.name=cobre \
		dataset.feature_type=timeseries \
		train.device="cpu" train.num_threads=1 \
		train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=4 \
		train.epochs=5

custom_gtn:
	python -m neurograph.train \
		check_commit=false \
		log.wandb_mode=disabled \
		model=custom_gtn \
		+model.timestep_model_config=sub_fedformer \
		model.timestep_model_config.attn_block_type='FEAw' \
		dataset.data_type=dense dataset.name=cobre \
		dataset.feature_type=timeseries \
		train.device="cpu" train.num_threads=1 \
		train.optim_args.weight_decay=0.1 \
		train.select_best_metric=loss train.num_outer_folds=4 \
		train.epochs=1

bgbGCN_random_crop:
	python -m neurograph.train \
	log.wandb_mode=disabled check_commit=false \
	dataset.data_path=/home/gl/skoltech/imaging/neurograph/test_datasets \
	dataset.data_type=graph dataset.name=hcp \
	dataset.atlas=shen \
	dataset.time_series_length=70 dataset.random_crop=true \
	dataset.random_crop_strategy='uniform' \
	model=bgbGCN \
	model.n_classes=7 \
	train.device="cpu" train.num_threads=1 train.device=cpu \
	train.select_best_metric=loss train.num_outer_folds=4 \
	train.epochs=1

bolt_morph:
	python -m neurograph.train \
		log.wandb_mode=disabled check_commit=false \
		train=bolt_train \
		model=multimodal_morph_bolt \
		model.dim=116 \
		model.n_classes=2 model.fusion_type=concat model.fusion_dim=1 \
		model.fusion_dropout=0.4 \
		dataset=morph_multimodal_dataset \
		dataset.name=cobre \
		dataset.fmri_feature_type=timeseries \
		dataset.time_series_length=150 \
		train.num_outer_folds=10 \
		train.batch_size=16 \
		train.valid_batch_size=32 \
		train.device=cpu \
		train.epochs=1 \
		train.select_best_model=true \
		train.select_best_metric=f1_macro train.patience=5

