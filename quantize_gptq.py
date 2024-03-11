from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache/")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir="cache/")

quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize="model.layers", model_seqlen=2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

quantizer.save(quantized_model, f'outputs/{model_name}/gptq/')
