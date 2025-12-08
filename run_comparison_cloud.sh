#!/bin/bash
set -e

# Setup environment
source venv/bin/activate

# 1. Run Text-Only Experiment
echo "Running Text-Only Experiment..."
python plot_cosine_similarity.py \
    --model_alias Qwen3-VL-8B-Instruct \
    --prompts_file test_prompts.json

# 2. Run Visual Experiment
echo "Running Visual Experiment (ignoring image tokens in plot)..."
python plot_cosine_similarity.py \
    --model_alias Qwen3-VL-8B-Instruct \
    --prompts_file test_prompts_visual.json \
    --image_file drugs.png

echo "Comparison experiments complete. Check measurements/cosine_similarity/ for output maps."
