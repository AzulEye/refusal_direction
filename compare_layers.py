
import json
import os

models = [
    ("Qwen-1.8B-Chat", "pipeline/runs/qwen-1_8b-chat"),
    ("Qwen3-VL-8B-Instruct", "pipeline/runs/Qwen3-VL-8B-Instruct")
]

for alias, path in models:
    metadata_path = os.path.join(path, "select_direction/direction_metadata.json")
    evals_path = os.path.join(path, "select_direction/direction_evaluations.json")
    
    print(f"\n--- {alias} ---")
    
    # 1. Get Selected Layer/Pos
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            # Support both formats (some might be list, some dict)
            if isinstance(meta, list): meta = meta[0] # Should be dict
            
            sel_layer = meta.get('layer')
            sel_pos = meta.get('position')
            print(f"Selected Direction: Layer {sel_layer}, Position {sel_pos}")
    else:
        print("Metadata not found.")
        sel_layer = None
        
    # 2. Get Total Layers
    if os.path.exists(evals_path):
        with open(evals_path, 'r') as f:
            evals = json.load(f)
            # Find max layer index
            max_layer = max(e['layer'] for e in evals)
            total_layers = max_layer + 1
            print(f"Total Layers: {total_layers}")
            
            if sel_layer is not None:
                ratio = sel_layer / total_layers
                print(f"Relative Depth: {ratio:.2%} (Layer {sel_layer} of {total_layers})")
    else:
        print("Evaluations not found.")
