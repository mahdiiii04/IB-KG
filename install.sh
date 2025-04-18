#!/bin/bash

echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo "Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

echo "Installing Transformers..."
pip install transformers

echo "Installing Jericho..."
pip install jericho

echo "Installing PyVis..."
pip install pyvis

echo "✅ All packages installed successfully."
