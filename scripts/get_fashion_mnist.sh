#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="/workspace/data/FashionMNIST/raw"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Fashion-MNIST test set (raw IDX files)
IMG_URL="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
LBL_URL="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"

if [ ! -f t10k-images-idx3-ubyte ]; then
  wget -q --show-progress "$IMG_URL" -O t10k-images-idx3-ubyte.gz || wget "$IMG_URL" -O t10k-images-idx3-ubyte.gz
  gunzip -f t10k-images-idx3-ubyte.gz
fi

if [ ! -f t10k-labels-idx1-ubyte ]; then
  wget -q --show-progress "$LBL_URL" -O t10k-labels-idx1-ubyte.gz || wget "$LBL_URL" -O t10k-labels-idx1-ubyte.gz
  gunzip -f t10k-labels-idx1-ubyte.gz
fi

echo "FashionMNIST test set ready at $DATA_DIR"
