from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer
import torch

model_name = "mistralai/Mistral-7B-v0.1"

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache/")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="cache/",
    quantization_config=quantization_config,
    device_map="auto"
)

