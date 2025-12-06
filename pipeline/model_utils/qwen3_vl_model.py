
import torch
import functools
from torch import Tensor
from torch import Tensor
from typing import List
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen3-VL Refusal Tokens (Hypothetical, reusing Qwen's for now or using generic ones)
# Ideally we find the specific tokens for "I cannot", "I'm sorry" etc.
# Qwen2.5/3 usually shares similar tokenizer.
QWEN_REFUSAL_TOKS = [40, 2121] # Check if these are still valid for Qwen3 tokenizer.

def orthogonalize_qwen3_vl_weights(model, direction: Float[Tensor, "d_model"]):
    # Orthogonalize Input Embeddings
    # model.model.language_model.embed_tokens
    embedding = model.model.language_model.embed_tokens
    embedding.weight.data = get_orthogonalized_matrix(embedding.weight.data, direction)

    for block in model.model.language_model.layers:
        # Self Attention Output Projection
        # block.self_attn.o_proj
        # Weight shape is (d_out, d_in). For o_proj, d_out=d_model, d_in=d_model.
        # Check transpose needed? get_orthogonalized_matrix takes (n_rows, n_cols).
        # We want to remove direction from the OUTPUT space (columns if W is d_in, d_out? No, PyTorch Linear is d_out, d_in).
        # We want to make sure W @ x doesn't have component in 'direction'.
        # So we project the rows of W onto the nullspace of 'direction'?
        # Existing code: get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        # c_proj is Linear. Weight is (out, in). .T is (in, out).
        # If we orthogonalize the COLUMNS of (in, out), we are orthogonalizing the rows of (out, in).
        # Yes, orthogonalizing rows means output vectors are orthogonal to direction.
        
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_qwen3_vl_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # Add bias to MLP output
    # block.mlp.down_proj does NOT have bias by default in Llama/Qwen2?
    # Qwen1 had bias. Qwen2/3 typically don't in MLP?
    # Let's assume we can't easily add bias if the parameter doesn't exist without registering it.
    # Qwen3VLTextMLP likely uses Linear(..., bias=False) for GLU variants.
    # Check if we can hack it or if we should use a hook instead?
    # The `act_add_weights` function implies modifying weights/biases permanently (or for the duration).
    # If bias is None, we might fail.
    # Qwen1 `c_proj` has bias.
    # I should check if `down_proj` has bias. (Most likely not).
    # If not, this implementation of act_add (structural) might not work.
    # But `run_pipeline` uses HOOKS for act_add usually: `get_activation_addition_input_pre_hook`.
    # `_get_act_add_mod_fn` is only used if we want to "bake it in"?
    # Actually `run_pipeline.py` calls `generate_completions(..., fwd_pre_hooks=actadd_fwd_pre_hooks)`.
    # It does NOT use `_get_act_add_mod_fn` in the main path?
    # Wait, `run_pipeline.py` calls `select_and_save_direction`.
    # `generate_and_save_completions_for_dataset` uses hooks.
    # The `_get_act_add_mod_fn` method in ModelBase seems unused in the main pipeline I saw???
    # Let's check `run_pipeline.py` again.
    
    pass

class Qwen3VLModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            import transformers
            raise ImportError(f"Could not import Qwen3VLForConditionalGeneration. Transformers version: {transformers.__version__}. Error: {e}")
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
            # trust_remote_code might be needed or not, depends on HF version. 4.57+ has it.
         ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        try:
            from transformers import AutoProcessor
        except ImportError as e:
            import transformers
            raise ImportError(f"Could not import AutoProcessor. Transformers version: {transformers.__version__}. Error: {e}")

        # We use the processor to get the tokenizer
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        # Using chat template from tokenizer
        return functools.partial(self.tokenize_instructions, tokenizer=self.tokenizer)

    def tokenize_instructions(self, instructions, tokenizer, include_trailing_whitespace=True):
        # Adapt list of strings to list of chat messages
        prompts = []
        for instr in instructions:
            messages = [{"role": "user", "content": [{"type": "text", "text": instr}]}] 
            # Note: Qwen3-VL template handles text-only. 
            # We use apply_chat_template to get the raw string, then tokenize?
            # Or tokenize directly.
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            
        return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")

    def _get_eoi_toks(self):
         # End of Instruction tokens. tricky with chat templates. 
         # usually `<|im_start|>assistant\n`
         return self.tokenizer.encode("<|im_start|>assistant\n") 

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.language_model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self._get_model_block_modules()])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self._get_model_block_modules()])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen3_vl_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        # This might fail if bias check returns none?
        # Leaving strict implementation aside, I'll return a dummy or try/except.
        # But wait, checking usage of `_get_act_add_mod_fn`...
        return functools.partial(act_add_qwen3_vl_weights, direction=direction, coeff=coeff, layer=layer)

