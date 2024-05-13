# pass `cu113` or `cpu` as the first arg
TORCH=1.12.1
DEVICE=$1

pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${DEVICE}

# use no-index so we just download whl files from the specified dir
pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${DEVICE}.html
pip install torch-geometric==2.0.3