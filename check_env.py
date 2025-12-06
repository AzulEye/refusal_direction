
import sys
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
    print(f"Transformers path: {transformers.__file__}")
except ImportError as e:
    print(f"Transformers: NOT INSTALLED ({e})")

try:
    from transformers import AutoProcessor
    print("AutoProcessor: OK")
except ImportError as e:
    print(f"AutoProcessor: FAIL ({e})")

try:
    from transformers import Qwen3VLForConditionalGeneration
    print("Qwen3VLForConditionalGeneration: OK")
except ImportError as e:
    print(f"Qwen3VLForConditionalGeneration: FAIL ({e})")
