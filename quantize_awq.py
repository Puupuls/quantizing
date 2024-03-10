from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = 'mistralai/Mistral-7B-v0.1'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(f'{model_name}/awq/')
tokenizer.save_pretrained(f'{model_name}/awq/')
