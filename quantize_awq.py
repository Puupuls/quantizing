from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = 'mistralai/Mistral-7B-v0.1'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name, cache_dir="cache/")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="cache/")

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(f'outputs/{model_name}/awq/')
tokenizer.save_pretrained(f'outputs/{model_name}/awq/')
