#!/bin/bash
set -e

# Setup environment
# Check if venv exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: venv/bin/activate not found. Assuming environment is already set up or trying global python."
fi

# Install missing dependencies
pip install -r requirements.txt

# 1. Run Text-Only Experiment
echo "Running Text-Only Experiment..."
python plot_cosine_similarity.py \
    --model_alias Qwen3-VL-8B-Instruct \
    --prompts_file test_prompts.json

# 2. Run Visual Experiment (Harmful)
echo "Running Visual Experiment (Harmful)..."
python plot_cosine_similarity.py \
    --model_alias Qwen3-VL-8B-Instruct \
    --prompts_file test_prompts_visual.json \
    --image_file drugs.png

# 3. Run Visual Experiment (Benign)
echo "Running Visual Experiment (Benign)..."
# Using test_prompts_visual_benign.json
python plot_cosine_similarity.py \
    --model_alias Qwen3-VL-8B-Instruct \
    --prompts_file test_prompts_visual_benign.json \
    --image_file drugs.png

echo "Comparison experiments complete. Check measurements/cosine_similarity/ for output maps."
