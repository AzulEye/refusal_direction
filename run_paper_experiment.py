
import os
import torch
import random
import numpy as np
from pipeline.config import Config
from pipeline.run_pipeline import (
    load_and_sample_datasets,
    select_and_save_direction,
    generate_and_save_candidate_directions,
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook

# Auto-detection in evaluate_jailbreak.py will select 8B model on GPU
# os.environ["QWEN3_GUARD_MODEL_ID"] = "Qwen/Qwen3Guard-Gen-8B"

def run_paper_experiment():
    # 1. Configuration
    cfg = Config(
        model_alias="Qwen3-VL-8B-Instruct",
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        n_train=128,
        n_test=100,  # Full size evaluation for paper
        n_val=32,
    )
    
    print(f"Starting Paper Experiment for {cfg.model_alias}...")
    print(f"Using Guardian: Auto-detected (Check logs for 8B/0.6B selection)")
    
    # 2. Load Model
    model_base = construct_model_base(cfg.model_path)
    
    # 3. Load Data
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    
    # 4. Generate & Select Direction
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    pos, layer, refusal_dir = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)
    
    print(f"Selected Refusal Direction: Layer {layer}, Position {pos}")

    # 5. Generate Completions & Evaluate
    
    # Intervention Hooks
    # Baseline: None
    # Ablation: (Handled internally by generate_completions logic? No, usually passed explicitly)
    # Wait, fit_and_save_direction returns the direction tensor.
    # run_pipeline.py logic:
    
    # A. Baseline
    print("Generating Baseline Completions...")
    generate_and_save_completions_for_dataset(
        cfg, model_base, fwd_pre_hooks=[], fwd_hooks=[], 
        intervention_label="baseline", dataset_name="jailbreakbench"
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg, intervention_label="baseline", dataset_name="jailbreakbench", 
        eval_methodologies=["substring_matching", "qwen3guard"]
    )
    
    # B. Ablation (Removal)
    # Get ablation hooks
    from pipeline.utils.hook_utils import get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook
    
    # We need to construct hooks carefully.
    # run_pipeline.py usually calls `get_direction_ablation_input_pre_hook` inside loops?
    # Let's check `run_pipeline.py` lines 103+.
    # Actually `run_pipeline.py` uses `evaluate_loss` for verification but for completions it does:
    
    # ablation_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=refusal_dir))]
    # ablation_fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=refusal_dir)), ...]
    
    # I should verify exact hook construction logic.
    # For now, I'll reconstruct it based on previous successful edits.
    
    if hasattr(model_base.model.config, "text_config"):
        n_layers_model = model_base.model.config.text_config.num_hidden_layers
    else:
        n_layers_model = model_base.model.config.num_hidden_layers
        
    ablation_fwd_pre_hooks = []
    ablation_fwd_hooks = []
    
    # Apply to all layers or just the selected one?
    # Previous code in `select_direction` applied to ALL layers for *candidate* generation?
    # No, `get_refusal_scores` uses specific hooks.
    # Standard "Refusal Direction" method usually ablates at ALL layers or specific one? 
    # Paper usually implies constructing a single direction and ablating it everywhere or at specific layer.
    # `hook_utils.py` `get_all_direction_ablation_hooks` applies to ALL layers.
    from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
    
    ablation_pre, ablation_fwd = get_all_direction_ablation_hooks(model_base, refusal_dir)
    
    print("Generating Ablation Completions...")
    generate_and_save_completions_for_dataset(
        cfg, model_base, fwd_pre_hooks=ablation_pre, fwd_hooks=ablation_fwd,
        intervention_label="ablation", dataset_name="jailbreakbench"
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg, intervention_label="ablation", dataset_name="jailbreakbench",
        eval_methodologies=["substring_matching", "qwen3guard"]
    )
    
    # C. ActAdd (Refusal Induction / Jailbreak Mitigation?)
    # For harmful prompts, ActAdd (Subtraction) should behave like Ablation (Jailbreak).
    # Coeff = -1.0
    actadd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=refusal_dir, coeff=-1.0))]
    
    print("Generating ActAdd (Subtract) Completions...")
    generate_and_save_completions_for_dataset(
        cfg, model_base, fwd_pre_hooks=actadd_hooks, fwd_hooks=[],
        intervention_label="actadd", dataset_name="jailbreakbench"
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg, intervention_label="actadd", dataset_name="jailbreakbench",
        eval_methodologies=["substring_matching", "qwen3guard"]
    )
    
    # D. Harmless Refusal Induction (Add)
    # Coeff = +1.0
    harmless_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=refusal_dir, coeff=1.0))]
    
    print("Generating Harmless ActAdd (Add) Completions...")
    generate_and_save_completions_for_dataset(
        cfg, model_base, fwd_pre_hooks=harmless_hooks, fwd_hooks=[],
        intervention_label="actadd", dataset_name="harmless"
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg, intervention_label="actadd", dataset_name="alpaca",
        eval_methodologies=["substring_matching"]
    )
    
    print("Paper Experiment Run Complete!")

if __name__ == "__main__":
    run_paper_experiment()
