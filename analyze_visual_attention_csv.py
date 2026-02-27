"""
CSV-based Visual Token Attention Analysis.

This is a simplified variant of analyze_visual_attention.py that reads rows
from a CSV with columns: prompt,image

Each row is treated as one sample (single image + single prompt).

Usage:
    source venv/bin/activate
    python analyze_visual_attention_csv.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --csv_path attacks_replace_with_object_prompts.csv \
        --output_dir measurements/visual_attention_csv
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def resolve_image_path(csv_path: str, image_value: str) -> str:
    """Resolve image path from CSV value (supports absolute or CSV-relative)."""
    image_value = str(image_value).strip()
    if os.path.isabs(image_value):
        return image_value
    return os.path.abspath(os.path.join(os.path.dirname(csv_path), image_value))


def load_csv_examples(csv_path, max_rows=None, prompt_contains=None):
    """Load prompt/image rows from CSV and validate image paths."""
    examples = []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"prompt", "image"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required_cols)}; "
                f"found {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=1):
            prompt = str(row.get("prompt", "")).strip()
            image_path = resolve_image_path(csv_path, row.get("image", ""))

            if not prompt or not image_path:
                skipped += 1
                continue
            if prompt_contains and prompt_contains.lower() not in prompt.lower():
                continue
            if not os.path.isfile(image_path):
                skipped += 1
                continue

            examples.append(
                {
                    "row_index": i,
                    "sample_id": f"row_{i}",
                    "prompt": prompt,
                    "image_path": image_path,
                }
            )

            if max_rows and len(examples) >= max_rows:
                break

    print(f"Loaded {len(examples)} examples from {csv_path} (skipped={skipped})")
    return examples


def get_visual_token_mask(inputs):
    """
    Detect visual token positions from processor inputs.

    Returns a bool tensor of shape (seq_len,) with True on visual token indices.
    """
    input_ids = inputs["input_ids"]

    if "image_grid_thw" not in inputs:
        return torch.zeros(input_ids.shape[1], dtype=torch.bool)

    grid = inputs["image_grid_thw"]
    n_visual = int(grid.prod(dim=-1).sum().item())
    ids = input_ids[0]

    unique, counts = ids.unique(return_counts=True)
    candidates = unique[counts == n_visual]
    if len(candidates) == 1:
        visual_token_id = candidates[0].item()
    else:
        visual_token_id = 151655  # Qwen image pad fallback

    return ids == visual_token_id


def analyze_attention(
    model,
    processor,
    examples,
    device,
    output_dir,
    max_image_size=256,
    csv_path=None,
):
    """
    Compute, per layer, attention mass from the last token to visual tokens.
    Aggregates over all CSV rows.
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model.config, "text_config"):
        n_layers = model.config.text_config.num_hidden_layers
    else:
        n_layers = model.config.num_hidden_layers

    all_mean_attn = []
    all_max_attn = []
    kept_ids = []

    for i, ex in enumerate(examples, start=1):
        sample_id = ex["sample_id"]
        prompt = ex["prompt"]

        image = Image.open(ex["image_path"]).convert("RGB")
        if max_image_size:
            image.thumbnail((max_image_size, max_image_size), Image.LANCZOS)

        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        visual_mask = get_visual_token_mask(inputs)
        n_visual = int(visual_mask.sum().item())
        if n_visual == 0:
            print(f"  [{i}/{len(examples)}] {sample_id}: no visual tokens found, skipping")
            continue

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        layer_mean_attn = np.zeros(n_layers)
        layer_max_attn = np.zeros(n_layers)
        # Ensure boolean mask is on the same device as attention tensors.
        visual_mask = visual_mask.to(outputs.attentions[0].device)
        for layer_idx, attn in enumerate(outputs.attentions):
            last_tok_attn = attn[0, :, -1, :]
            visual_attn_per_head = last_tok_attn[:, visual_mask].sum(dim=-1)
            layer_mean_attn[layer_idx] = visual_attn_per_head.float().mean().item()
            layer_max_attn[layer_idx] = visual_attn_per_head.float().max().item()

        all_mean_attn.append(layer_mean_attn)
        all_max_attn.append(layer_max_attn)
        kept_ids.append(sample_id)

        print(
            f"  [{i}/{len(examples)}] {sample_id}: "
            f"{n_visual} visual toks, "
            f"peak mean @ layer {np.argmax(layer_mean_attn)} "
            f"({layer_mean_attn.max():.4f})"
        )

        del outputs, inputs
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_mean_attn:
        raise RuntimeError("No examples produced valid attention data.")

    mean_curve = np.mean(all_mean_attn, axis=0)
    max_curve = np.mean(all_max_attn, axis=0)
    std_mean = np.std(all_mean_attn, axis=0)
    n = len(all_mean_attn)
    sem_mean = std_mean / np.sqrt(n)

    result = {
        "csv_path": os.path.abspath(csv_path) if csv_path else None,
        "n_examples": n,
        "n_layers": int(n_layers),
        "mean_attention_to_visual": mean_curve.tolist(),
        "max_attention_to_visual": max_curve.tolist(),
        "sem_mean": sem_mean.tolist(),
        "example_ids": kept_ids,
        "per_example_mean": [a.tolist() for a in all_mean_attn],
        "per_example_max": [a.tolist() for a in all_max_attn],
    }

    json_path = os.path.join(output_dir, "attention_curve.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved attention data to {json_path}")

    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, mean_curve, color="tab:blue", linewidth=2, label="Mean across heads")
    ax.fill_between(
        layers,
        mean_curve - 1.96 * sem_mean,
        mean_curve + 1.96 * sem_mean,
        color="tab:blue",
        alpha=0.2,
    )
    ax.plot(
        layers,
        max_curve,
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        label="Max head (avg across examples)",
    )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Attention fraction to visual tokens", fontsize=13)
    ax.set_title("Visual Token Attention by Layer (CSV samples)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_layers - 1)

    plot_path = os.path.join(output_dir, "attention_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze visual token attention across layers using a prompt,image CSV"
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV path with columns: prompt,image")
    parser.add_argument("--output_dir", type=str, default="measurements/visual_attention_csv")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit number of CSV rows (for quick runs)")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="Max image dimension (thumbnail size)")
    parser.add_argument("--prompt_contains", type=str, default=None,
                        help="Optional substring filter on prompt text")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/mps/cpu)")
    parsed_args = parser.parse_args()

    if parsed_args.device:
        device = torch.device(parsed_args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    examples = load_csv_examples(
        parsed_args.csv_path,
        max_rows=parsed_args.max_rows,
        prompt_contains=parsed_args.prompt_contains,
    )
    if not examples:
        raise RuntimeError("No valid CSV rows found.")

    print(f"Loading model: {parsed_args.model_path}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        parsed_args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(parsed_args.model_path)

    analyze_attention(
        model,
        processor,
        examples,
        device,
        parsed_args.output_dir,
        max_image_size=parsed_args.max_image_size,
        csv_path=parsed_args.csv_path,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
