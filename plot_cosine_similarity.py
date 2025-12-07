
import os
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from tqdm import tqdm
import transformers
import sys
from types import ModuleType

# Mock transformers_stream_generator to avoid ImportError due to transformers mismatch
# This module is optional for Qwen but its presence causes crash if incompatible
try:
    import transformers_stream_generator
except ImportError:
    # If it fails to import (which it does), we mock it.
    # Actually, the internal import fails. So we pre-emptively mock it.
    pass

# Force mock
dummy_gen = ModuleType("transformers_stream_generator")
dummy_gen.init_stream_support = lambda: None
sys.modules["transformers_stream_generator"] = dummy_gen

# Also patch BeamSearchScorer just in case
try:
    from transformers.generation.beam_search import BeamSearchScorer
    transformers.BeamSearchScorer = BeamSearchScorer
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

def get_activations_hook(layer, cache):
    def hook(module, input):
        # input is tuple (tensor, ) for pre-hook
        # output is NOT available in pre-hook

        # input is tuple, output is tensor (batch, seq, d_model)
        # We capture output of the block (residual stream)
        # cache[layer] = output.detach().cpu()
        
        # If input[0] is the residual stream BEFORE the block? 
        # Standard practice is usually "resid_post" (output of block) or "resid_pre" (input to block).
        # Our direction was trained on ... usually resid_post or resid_pre?
        # SelectDirection uses `model_base.model_block_modules`.
        # Hook is `get_direction_ablation_input_pre_hook` -> PRE hook.
        # So it modifies input to the block.
        # This implies the direction exists in the residual stream entering the block. (Resid_Pre).
        # So we should capture input[0].
        
        if isinstance(input, tuple):
            act = input[0]
        else:
            act = input
            
        cache[layer] = act.detach().cpu()
    return hook

def plot_cosine_similarity(args):
    # 1. Load Model
    print(f"Loading model: {args.model_alias}")
    # We use construct_model_base to handle Qwen3-VL specifics
    # But we need to know the path. Assuming standard path or passed in args.
    # If alias is passed, we might need mapping.
    # For now, let's assume alias IS the path or we use the config map.
    
    # If alias matches directory name, we try to deduce common HF paths or use mapping.
    # User Run Directory: "qwen-1_8b-chat" -> HF: "Qwen/Qwen-1_8B-Chat"
    # User Run Directory: "Qwen3-VL-8B-Instruct" -> HF: "Qwen/Qwen3-VL-8B-Instruct"
    
    # We can try to construct path from alias if not in map.
    model_paths = {
        "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen-1.8B-Chat": "Qwen/Qwen-1_8B-Chat",
        "qwen-1_8b-chat": "Qwen/Qwen-1_8B-Chat", # Handle the directory name case
        "gemma-2b-it": "google/gemma-2b-it"
    }
    path = model_paths.get(args.model_alias)
    if not path:
        # Heuristic: If alias contains "Qwen", assume "Qwen/" prefix?
        # Better: just use alias as path if checks pass, or assume mapped.
        # For this task, strict mapping is safer.
        print(f"Warning: No explicit mapping for {args.model_alias}. Using as path.")
        path = args.model_alias
    
    model_base = construct_model_base(path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    
    # 2. Load Direction
    possible_paths = [
        f"pipeline/runs/{args.model_alias}/select_direction/direction.pt",
        f"pipeline/runs/{args.model_alias}/direction.pt"
    ]
    direction_path = None
    for p in possible_paths:
        if os.path.exists(p):
            direction_path = p
            break
            
    if not direction_path:
        raise FileNotFoundError(f"Could not find direction.pt in {possible_paths}")
        
    print(f"Loading direction from: {direction_path}")
    direction = torch.load(direction_path, map_location='cpu').float()
    direction = direction / direction.norm() # Ensure unit vector
    
    # 3. Load Prompts
    if args.prompts_file.endswith('.json'):
        with open(args.prompts_file, 'r') as f:
            data = json.load(f)
            # Support list of strings or list of dicts
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    prompts = data
                elif data and isinstance(data[0], dict):
                    # Try common keys
                    prompts = []
                    for item in data:
                        if 'prompt' in item: prompts.append(item['prompt'])
                        elif 'instruction' in item: prompts.append(item['instruction'])
                        elif 'text' in item: prompts.append(item['text'])
                else:
                    raise ValueError("JSON must be a list of strings or a list of dicts with 'prompt'/'instruction' keys.")
            else:
                 raise ValueError("JSON must be a list.")
    else:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
    # 4. Setup Hooks
    if hasattr(model.config, "text_config"):
        n_layers = model.config.text_config.num_hidden_layers
    else:
        n_layers = model.config.num_hidden_layers
        
    # 5. Process Prompts
    output_dir = "measurements/cosine_similarity"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:30]}...")
        
        # Use model_base abstraction for tokenization which handles different models
        inputs = model_base.tokenize_instructions_fn(instructions=[prompt])
        inputs = inputs.to(model.device)
        
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]
        
        # Run forward pass with hooks
        cache = {}
        hooks = []
        for layer in range(n_layers):
            # Capture Input to module (Resid_Pre)
            h = model_base.model_block_modules[layer].register_forward_pre_hook(get_activations_hook(layer, cache))
            hooks.append(h)
            
        with torch.no_grad():
            model(**inputs)
            
        # Remove hooks
        for h in hooks: h.remove()
        
        # Compute Cosine Similarity
        # Cache[layer]: (1, seq_len, d_model)
        heatmap_data = np.zeros((n_layers, len(tokens)))
        
        for layer in range(n_layers):
            acts = cache[layer][0].float() # (seq_len, d_model)
            acts = acts / acts.norm(dim=-1, keepdim=True)
            
            sim = (acts @ direction).numpy()
            heatmap_data[layer, :] = sim
            
        # Plot
        plt.figure(figsize=(20, 10))
        sns.heatmap(heatmap_data, cmap="RdBu_r", vmin=0, vmax=1, yticklabels=True) 
        
        plt.xlabel("Token Position")
        plt.ylabel("Layer")
        plt.title(f"Cosine Similarity with Refusal Direction\nPrompt: {prompt[:50]}...")
        
        # Set x-ticks to tokens (might be crowded)
        if len(tokens) < 100:
            plt.xticks(np.arange(len(tokens)) + 0.5, tokens, rotation=90, fontsize=8)
        else:
            step = len(tokens) // 20
            plt.xticks(np.arange(0, len(tokens), step) + 0.5, [tokens[j] for j in range(0, len(tokens), step)], rotation=90)
            
        filename = f"{output_dir}/{args.model_alias}_prompt_{i}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")
        
        # Generate Response
        print("Generating response...")
        # use_cache=False to avoid incompatibility with newer transformers and custom Qwen model code
        gen_out = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=False)
        # Decode only the new tokens
        new_tokens = gen_out[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Response: {response_text}\n" + "-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_alias", type=str, required=True, help="Model alias (e.g., Qwen3-VL-8B-Instruct)")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to text file with prompts")
    args = parser.parse_args()
    plot_cosine_similarity(args)
