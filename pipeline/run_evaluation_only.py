import os
import sys

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config
from pipeline.run_pipeline import evaluate_completions_and_save_results_for_dataset

def main():
    # We strictly need the config to know paths and methodologies
    # The config expects a model_path argument to init, but for artifacts path it uses model_alias.
    
    # Using the alias from the previous run
    cfg = Config(model_alias="Qwen-1_8B-Chat", model_path="Qwen/Qwen-1_8B-Chat")
    
    interventions = ['baseline', 'ablation', 'actadd']
    datasets = cfg.evaluation_datasets # ('jailbreakbench',)
    
    print(f"Eval methodologies: {cfg.jailbreak_eval_methodologies}")

    for dataset_name in datasets:
        for intervention in interventions:
            print(f"Evaluating {dataset_name} {intervention}...")
            try:
                evaluate_completions_and_save_results_for_dataset(
                    cfg, 
                    intervention, 
                    dataset_name, 
                    eval_methodologies=cfg.jailbreak_eval_methodologies
                )
            except Exception as e:
                print(f"Failed to evaluate {dataset_name} {intervention}: {e}")

    # Also harmless dataset
    print("Evaluating harmless...")
    try:
        evaluate_completions_and_save_results_for_dataset(
            cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies
        )
    except Exception as e:
        print(f"Failed to evaluate harmless: {e}")

if __name__ == "__main__":
    main()
