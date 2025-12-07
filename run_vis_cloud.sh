#!/bin/bash
set -e

# Ensure dependencies are installed
pip install -q seaborn matplotlib

echo "Running visualization for Qwen3-VL-8B-Instruct..."
python plot_cosine_similarity.py --model_alias Qwen3-VL-8B-Instruct --prompts_file test_prompts.json

echo "Done! Check measurements/cosine_similarity/ for output."
