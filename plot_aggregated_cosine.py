
import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_model_and_type(filename):
    basename = os.path.basename(filename)
    
    # Identify Model
    if basename.startswith("Qwen3-VL-8B-Instruct"):
        model = "Qwen3-VL-8B-Instruct"
    elif basename.startswith("Qwen3-VL-32B-Instruct"):
        model = "Qwen3-VL-32B-Instruct"
    else:
        return None, None
        
    # Identify Attack Type
    if "_naive_attack_" in basename:
        attack_type = "naive_attack"
    elif "_replace_with_object_" in basename:
        attack_type = "replace_with_object"
    else:
        return model, "unknown"
        
    return model, attack_type

def load_data(results_dir):
    data = {} # {model: {attack_type: [vectors]}}
    
    files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(files)} JSON files in {results_dir}")
    
    for f in files:
        model, attack_type = parse_model_and_type(f)
        if not model or not attack_type:
            continue
            
        if model not in data:
            data[model] = {}
        if attack_type not in data[model]:
            data[model][attack_type] = []
            
        try:
            with open(f, 'r') as fp:
                content = json.load(fp)
                
            heatmap = np.array(content['cosine_similarity']['heatmap'])
            # heatmap shape: (n_layers, n_tokens)
            
            # Take last 3 tokens
            if heatmap.shape[1] >= 3:
                last_3 = heatmap[:, -3:]
                # Mean over tokens
                vec = np.mean(last_3, axis=1) # (n_layers,)
                data[model][attack_type].append(vec)
            else:
                print(f"Skipping {f}: Not enough tokens ({heatmap.shape[1]})")
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return data

def plot_aggregated(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Style settings to match paper
    # sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })
    
    colors = {
        "naive_attack": "blue",
        "replace_with_object": "orange" 
        # Paper uses Blue for "harmful" (naive) and Orange for "variable" (random suffix).
        # Replacement is a variation, so Orange fits.
    }
    
    labels = {
        "naive_attack": "Naive Attack",
        "replace_with_object": "Replacement Attack"
    }

    for model, attack_data in data.items():
        plt.figure(figsize=(8, 6))
        
        has_data = False
        
        for attack_type, vectors in attack_data.items():
            if not vectors:
                continue
                
            has_data = True
            arr = np.array(vectors) # (N, n_layers)
            n_samples = arr.shape[0]
            
            mean_vec = np.mean(arr, axis=0)
            std_vec = np.std(arr, axis=0) # Standard Deviation
            sem_vec = std_vec / np.sqrt(n_samples) # Standard Error
            
            # 95% Confidence Interval
            ci = 1.96 * sem_vec
            
            layers = np.arange(len(mean_vec))
            
            # Plot
            color = colors.get(attack_type, "gray")
            label = f"{labels.get(attack_type, attack_type)} (N={n_samples})"
            
            plt.plot(layers, mean_vec, color=color, label=label, linewidth=2)
            plt.fill_between(layers, mean_vec - ci, mean_vec + ci, color=color, alpha=0.2)
            
        if has_data:
            plt.title(f"Cosine Similarity vs Layer ({model})")
            plt.xlabel("Layer")
            plt.ylabel("Cosine similarity with refusal direction")
            plt.legend()
            
            # Limit Y axis to reasonable range if needed, e.g., similar to paper 0 to 0.6
            # But let's let autoscaling handle it unless it's weird.
            # Paper goes -0.05 to 0.55
            
            outfile = os.path.join(output_dir, f"aggregated_cosine_{model}.png")
            plt.tight_layout()
            plt.savefig(outfile, dpi=300)
            print(f"Saved plot to {outfile}")
            plt.close()
        else:
            print(f"No valid data to plot for {model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="measurements/results_from_cloud")
    parser.add_argument("--output_dir", type=str, default="measurements/figures")
    args = parser.parse_args()
    
    print(f"Reading results from: {args.results_dir}")
    data = load_data(args.results_dir)
    plot_aggregated(data, args.output_dir)
