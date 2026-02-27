# ── 1. Clone ──
git clone https://github.com/AzulEye/refusal_direction.git && cd refusal_direction

# ── 2. Python env ──
python -m venv venv && source venv/bin/activate
pip install torch torchvision transformers accelerate pillow matplotlib tqdm numpy litellm gdown jaxtyping einops sentencepiece protobuf datasets

# ── 3. Download from Google Drive ──
gdown 1xt0P6FGP057Hp57TGZ7czpjUfeJ61BYR -O data_object_replacement_simple.zip
gdown 1ePQwvmdi8wkWZPu0kShK3PA_yOvbyLoB -O measurements.zip

# ── 4. Uncompress into the right places ──
unzip -o data_object_replacement_simple.zip -d dataset/
unzip -o measurements.zip -d .

# ── 5. Pull latest & run ──
git pull
mkdir -p measurements/visual_attention_32b_csv

python analyze_visual_attention_csv.py \
    --model_path Qwen/Qwen3-VL-32B-Instruct \
    --csv_path attacks_replace_with_object_prompts.csv \
    --data_root dataset \
    --output_dir measurements/visual_attention_32b_csv \
    --max_rows 0 --max_image_size 512 \
    --prompt_contains banana \
    2>&1 | tee measurements/visual_attention_32b_csv/attention_32b_attacks_banana_single.log

python layerwise_refusal_ablation_csv.py \
    --model_path Qwen/Qwen3-VL-32B-Instruct \
    --model_alias Qwen3-VL-32B-Instruct \
    --csv_path attacks_replace_with_object_prompts.csv \
    --data_root dataset \
    --output_dir measurements/visual_attention_32b_csv \
    --attention_json measurements/visual_attention_32b_csv/attention_curve.json \
    --max_rows 0 --max_image_size 512 \
    --layer_range 0 20 \
    --batch_size 8 \
    --verbose \
    --prompt_contains banana \
    2>&1 | tee measurements/visual_attention_32b_csv/32b_single_layer_sweep.log
