
import json
import glob
import os

def check_disagreements(filepath):
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    completions = data.get('completions', [])
    disagreements = []

    for i, item in enumerate(completions):
        sub_match = item.get('is_jailbreak_substring_matching')
        qwen_guard = item.get('is_jailbreak_qwen3guard')

        # We are looking for cases where they differ
        if sub_match != qwen_guard:
            disagreement_type = "Substring says Jailbreak, QwenGuard says Safe" if sub_match else "Substring says Safe, QwenGuard says Jailbreak"
            disagreements.append({
                'index': i,
                'prompt': item.get('prompt', '')[:100] + "...",
                'response': item.get('response', '')[:300] + "...", # Truncate for readability
                'substring_label': sub_match,
                'qwen_guard_label': qwen_guard,
                'type': disagreement_type
            })

    print(f"Found {len(disagreements)} disagreements.")
    
    # Show first 5 examples
    for d in disagreements[:5]:
        print("-" * 80)
        print(f"Type: {d['type']}")
        print(f"Prompt: {d['prompt']}")
        print(f"Response: {d['response']}")
        print(f"Substring Match: {d['substring_label']}")
        print(f"QwenGuard: {d['qwen_guard_label']}")

files = glob.glob('pipeline/runs/Qwen3-VL-8B-Instruct/completions/*_evaluations.json')
for f in files:
    check_disagreements(f)
