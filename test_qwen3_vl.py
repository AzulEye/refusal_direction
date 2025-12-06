from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# Minimal test script for Qwen3-VL
model_id = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Loading processor for {model_id}...")
# Use trust_remote_code=True if needed? The docs didn't specify, but often new models need it.
# Docs say "The code of Qwen3-VL has been in the latest Hugging Face transformers", so maybe not needed.
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded.")

print(f"Loading model for {model_id}...")
try:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to verify if it's just import error
    exit(1)

# Simple text-only test (if supported) or dummy image test
# Qwen3-VL expects images in the chat template mostly, but can handle text only?
# Let's try the example format but without actual image (or a public URL)
messages = [
    {
        "role": "user",
        "content": [
             # Use a very small placeholder image or just text if possible. 
             # The example uses an online image. We'll use the same one.
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

print("Preparing inputs...")
try:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    print("Inputs prepared.")
    
    print("Generating...")
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Output:", output_text)
    
except Exception as e:
    print(f"Inference failed: {e}")
