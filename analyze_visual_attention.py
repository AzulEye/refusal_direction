"""
Part 1: Visual Token Attention Analysis

For each layer in a Qwen3-VL model, measures how much attention the last
text token pays to visual tokens vs text tokens.  This reveals the layer
at which the model begins integrating visual information.

Usage:
    source venv/bin/activate
    python analyze_visual_attention.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --max_behaviors 10
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def _normalize_replacement_label(value):
    """Normalize replacement labels for robust matching."""
    return str(value).strip().lower().replace("_", " ")


def load_behaviors(data_root, max_behaviors=None, replacement_filter=None):
    """
    Load visual replacement behaviors from dataset.
    Selects only ONE image per slot variable (X1, X2, ...) to match
    a realistic single-attempt attack.

    Returns a list of dicts:
        {behavior_id, neutralized_prompt, image_paths}
    """
    results_dir = os.path.join(data_root, "results")
    attacks_dir = os.path.join(data_root, "data", "attacks")
    behaviors = []

    for bid in sorted(os.listdir(results_dir)):
        meta_path = os.path.join(results_dir, bid, "harmbench_neutralized", "metadata.json")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if replacement_filter:
            target = _normalize_replacement_label(replacement_filter)
            slot_repls = meta.get("slot_replacements", {}) or {}
            repl_values = [
                _normalize_replacement_label(v)
                for v in slot_repls.values()
                if v is not None
            ]
            if target not in repl_values:
                continue

        # Collect local attack images
        attack_dir = os.path.join(attacks_dir, bid)
        if not os.path.isdir(attack_dir):
            continue

        all_images = sorted(glob.glob(os.path.join(attack_dir, "*.png")))
        if not all_images:
            continue

        # Pick one image per slot (X1, X2, ...).
        # Filenames look like: X1_arson_attack_..._1.png, X1_..._2.png, X2_..._1.png
        # Group by slot prefix and take the first image from each.
        seen_slots = set()
        selected_images = []
        for img_path in all_images:
            fname = os.path.basename(img_path)
            slot = fname.split("_")[0]  # e.g. "X1", "X2"
            if slot not in seen_slots:
                seen_slots.add(slot)
                selected_images.append(img_path)

        behaviors.append({
            "behavior_id": bid,
            "neutralized_prompt": meta.get("neutralized_prompt", ""),
            "image_paths": selected_images,
        })

        if max_behaviors and len(behaviors) >= max_behaviors:
            break

    return behaviors


def get_visual_token_mask(inputs):
    """
    Determine which positions in input_ids are visual (image) tokens.

    Qwen3-VL uses special <|image_pad|> or <|vision_*|> tokens that can be
    identified by checking against the processor's image token id, or by
    looking at `image_grid_thw` to compute the expected count of visual tokens.

    Returns a boolean tensor of shape (seq_len,) — True for visual positions.
    """
    input_ids = inputs["input_ids"]  # (1, seq_len)

    if "image_grid_thw" in inputs:
        # Qwen3-VL uses token id 151655 for <|image_pad|> (the vision placeholder)
        # but the exact id depend on the tokenizer. We detect by finding the
        # contiguous block whose length matches the expected visual token count.
        #
        # image_grid_thw: (n_images, 3) — each row is (t, h, w).
        # Total visual tokens = sum(t * h * w) for each image.
        grid = inputs["image_grid_thw"]  # tensor
        n_visual = int(grid.prod(dim=-1).sum().item())

        # The visual tokens in Qwen3-VL are typically token id 151655 (<|image_pad|>)
        # We find them by looking for the most common non-text token in bulk.
        ids = input_ids[0]

        # Heuristic: the image_pad token appears exactly n_visual times.
        # Find token id that appears n_visual times.
        unique, counts = ids.unique(return_counts=True)
        candidates = unique[counts == n_visual]
        if len(candidates) == 1:
            visual_token_id = candidates[0].item()
        else:
            # Fallback: use known Qwen3-VL image_pad token id
            visual_token_id = 151655

        mask = (ids == visual_token_id)
        return mask
    else:
        # No images — everything is text
        return torch.zeros(input_ids.shape[1], dtype=torch.bool)


def analyze_attention(model, processor, behaviors, device, output_dir, max_image_size=256):
    """
    For each behavior, run a forward pass with output_attentions=True.
    Compute per-layer fraction of attention from the last text token to
    visual tokens.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_layers = model.config.text_config.num_hidden_layers
    # Accumulate per-layer attention-to-visual for averaging and max
    all_mean_attn = []  # list of (n_layers,) arrays  — mean across heads
    all_max_attn = []   # list of (n_layers,) arrays  — max across heads

    for bi, beh in enumerate(behaviors):
        bid = beh["behavior_id"]
        prompt = beh["neutralized_prompt"]
        img_paths = beh["image_paths"]

        # Load and resize images to limit visual token count
        images = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if max_image_size:
                img.thumbnail((max_image_size, max_image_size), Image.LANCZOS)
            images.append(img)

        # Build chat input
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")

        # Identify visual token positions
        visual_mask = get_visual_token_mask(inputs)  # (seq_len,)
        n_visual = visual_mask.sum().item()
        if n_visual == 0:
            print(f"  [{bi+1}] {bid}: no visual tokens found, skipping")
            continue

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of (batch, n_heads, seq_len, seq_len) per layer
        layer_mean_attn = np.zeros(n_layers)
        layer_max_attn = np.zeros(n_layers)

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (1, n_heads, seq_len, seq_len)
            # We want: attention FROM the last token TO visual tokens
            last_tok_attn = attn[0, :, -1, :]  # (n_heads, seq_len)

            # Fraction of attention to visual tokens per head
            visual_attn_per_head = last_tok_attn[:, visual_mask].sum(dim=-1)  # (n_heads,)
            # (attention weights already sum to 1 over seq_len dim)

            layer_mean_attn[layer_idx] = visual_attn_per_head.float().mean().item()
            layer_max_attn[layer_idx] = visual_attn_per_head.float().max().item()

        all_mean_attn.append(layer_mean_attn)
        all_max_attn.append(layer_max_attn)

        print(f"  [{bi+1}/{len(behaviors)}] {bid}: "
              f"{n_visual} visual toks, "
              f"peak mean attn @ layer {np.argmax(layer_mean_attn)} "
              f"({layer_mean_attn.max():.4f})")

        # Free memory
        del outputs, inputs
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_mean_attn:
        print("ERROR: No behaviors produced valid attention data.")
        return

    # Aggregate across behaviors
    mean_curve = np.mean(all_mean_attn, axis=0)  # (n_layers,)
    max_curve = np.mean(all_max_attn, axis=0)     # avg of per-behavior max-head
    std_mean = np.std(all_mean_attn, axis=0)
    n = len(all_mean_attn)
    sem_mean = std_mean / np.sqrt(n)

    # Save raw data
    result = {
        "n_behaviors": n,
        "n_layers": int(n_layers),
        "mean_attention_to_visual": mean_curve.tolist(),
        "max_attention_to_visual": max_curve.tolist(),
        "sem_mean": sem_mean.tolist(),
        "per_behavior_mean": [a.tolist() for a in all_mean_attn],
        "per_behavior_max": [a.tolist() for a in all_max_attn],
    }

    json_path = os.path.join(output_dir, "attention_curve.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved attention data to {json_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(n_layers)

    ax.plot(layers, mean_curve, color="tab:blue", linewidth=2, label="Mean across heads")
    ax.fill_between(layers, mean_curve - 1.96 * sem_mean, mean_curve + 1.96 * sem_mean,
                     color="tab:blue", alpha=0.2)
    ax.plot(layers, max_curve, color="tab:orange", linewidth=2, linestyle="--",
            label="Max head (avg across behaviors)")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Attention fraction to visual tokens", fontsize=13)
    ax.set_title("Visual Token Attention by Layer (last token → visual tokens)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_layers - 1)

    plot_path = os.path.join(output_dir, "attention_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze visual token attention across layers")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--data_root", type=str, default="dataset/visual_replacement")
    parser.add_argument("--output_dir", type=str, default="measurements/visual_attention")
    parser.add_argument("--max_behaviors", type=int, default=10,
                        help="Max number of behaviors to process (for fast iteration)")
    parser.add_argument("--max_image_size", type=int, default=256,
                        help="Max image dimension (shorter side). Smaller = fewer visual tokens.")
    parser.add_argument("--replacement_filter", type=str, default=None,
                        help="Keep only behaviors whose slot_replacements include this object (e.g. banana)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detected if not set)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {args.model_path}")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Required for output_attentions=True
    ).eval()
    model.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Model loaded. LM layers: {model.config.text_config.num_hidden_layers}")

    # Load behaviors
    behaviors = load_behaviors(
        args.data_root,
        max_behaviors=args.max_behaviors,
        replacement_filter=args.replacement_filter,
    )
    repl_desc = f" (replacement filter='{args.replacement_filter}')" if args.replacement_filter else ""
    print(f"Loaded {len(behaviors)} behaviors{repl_desc}")

    if not behaviors:
        print("ERROR: No behaviors found. Check --data_root path.")
        return

    # Run analysis
    analyze_attention(model, processor, behaviors, device, args.output_dir,
                      max_image_size=args.max_image_size)
    print("\nDone!")


if __name__ == "__main__":
    main()
