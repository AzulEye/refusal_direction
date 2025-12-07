#!/bin/bash
set -e

# Ensure dependencies are installed
pip install -q seaborn matplotlib
# Fix for PIL.Image.Resampling error (requires Pillow >= 9.1.0)
pip install --upgrade pillow
# Ensure latest transformers for FP8 support
pip install --upgrade transformers accelerate

# Switching back to standard model (BF16) as FP8 vision encoder crashes
MODEL_ALIAS="Qwen3-VL-32B-Instruct"
MODEL_PATH="Qwen/Qwen3-VL-32B-Instruct"

echo "Running paper experiment for $MODEL_ALIAS..."
python run_paper_experiment.py --model_alias "$MODEL_ALIAS" --model_path "$MODEL_PATH"

echo "Done! Check pipeline/runs/$MODEL_ALIAS/ for output."
