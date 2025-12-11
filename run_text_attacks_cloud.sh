#!/bin/bash

# Configuration
# Using Qwen3-VL-8B-Instruct as standard, can switch to 32B if needed
MODEL_ALIAS="Qwen3-VL-8B-Instruct" 

# Directory containing the text attack JSONs 
# (Assumed to be pulled from git to measurements/text_attacks)
TEXT_ATTACK_DIR="measurements/text_attacks"

# 1. Run plot_cosine_similarity.py in batch mode
# It will recursively find all JSONs in TEXT_ATTACK_DIR and process them.
# The script is smart enough to handle text_harmful / text_replacement types.

echo "Starting Text Attack Batch Processing..."
python plot_cosine_similarity.py \
    --model_alias "$MODEL_ALIAS" \
    --batch_dir "$TEXT_ATTACK_DIR" \
    --max_new_tokens 512 \
    --max_images 1 

echo "Batch processing complete."

# 2. Run Aggregation Plot (Optional, but useful to verify results immediately)
echo "Generating Aggregated Plots..."
python plot_aggregated_cosine.py 

echo "Done. Results are in measurements/cosine_similarity/ and measurements/plots/"
