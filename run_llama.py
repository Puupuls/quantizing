import gc
import shutil
from csv import DictWriter
from time import time

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from controllers.ConrollerQuantizer import ControllerQuantizer
from controllers.ControllerEvaluator import ControllerEvaluator
from utils import get_gpu_utilization

gpu_background_ram = get_gpu_utilization()
model_name = "Undi95/Meta-Llama-3-8B-hf"

load_start = time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
load_end = time()
gpu_loaded = get_gpu_utilization()

results_csv = model_name.replace("/",'_')+"--awq.csv"
results_obj = {}

fieldnames = [
    'base', 'eval_time', 'ds', 'ds_size', 'bits', 'model_seqlen', 'q_group_size', 'method', 'type', 'load_time',
    'gpu_after_loading', 'gpu_background_ram', 'gpu_quantized', 'quant_time', 'similarities', 'bleus', 'failures'
]
evaluator = ControllerEvaluator(tokenizer=tokenizer)

time_start = time()
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
results = evaluator.evaluate(model=model)
results_obj["original"] = {
    'base': model_name,
    "eval_time": time()-time_start,
    "ds": "HuggingFaceH4/no_robots test_sft",
    "ds_size": len(evaluator.ds),
    "bits": "N/A",
    "model_seqlen": "N/A",
    "q_group_size": "N/A",
    "method": "N/A",
    "type": "N/A",
    "load_time": load_end - load_start,
    "gpu_after_loading": gpu_loaded,
    "gpu_background_ram": gpu_background_ram,
    "gpu_quantized": "N/A",
    "quant_time": "N/A",
    **results
}
with open(results_csv, "w") as f:
    writer = DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(results_obj["original"])
del model
del results
torch.cuda.empty_cache()
gc.collect()
shutil.rmtree("./cache", ignore_errors=True)

# for i in [4]:  # only 4bit supported for now
for m in ["GEMV", "GEMM"]:  # GEMV is crashing
    model, metrics = ControllerQuantizer.awq(model_name, quant_config={"version": m, "w_bit": 4})
    time_start = time()
    results = evaluator.evaluate(model=model)
    results_obj[f"awq"] = {
        'base': model_name,
        "eval_time": time()-time_start,
        "ds": "HuggingFaceH4/no_robots test_sft",
        "ds_size": len(evaluator.ds),
        "type": "AWQ",
        "method": m,
        "bits": 4,
        "model_seqlen": "N/A",
        "q_group_size": "N/A",
        **metrics,
        **results
    }
    with open(results_csv, "a") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results_obj[f"awq"])

del model
del metrics
torch.cuda.empty_cache()
gc.collect()
shutil.rmtree("./cache", ignore_errors=True)

for bits in range(2, 8):
    for seq_len in [512, 1024, 2048]:
        for group_size in [256]:
            try:
                model, metrics = ControllerQuantizer.gptq(model_name, quant_config={
                    "bits": bits,
                    "model_seqlen": seq_len,
                    "q_group_size": group_size,
                })
                time_start = time()
                results = evaluator.evaluate(model=model)
                results_obj[f"gptq_{bits}bit_{seq_len}seq_{group_size}group"] = {
                    'base': model_name,
                    "eval_time": time()-time_start,
                    "ds": "HuggingFaceH4/no_robots test_sft",
                    "ds_size": len(evaluator.ds),
                    "type": "GPTQ",
                    "method": "N/A",
                    "bits": bits,
                    "model_seqlen": seq_len,
                    "q_group_size": group_size,
                    **metrics,
                    **results
                }
                with open(results_csv, "a") as f:
                    writer = DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(results_obj[f"gptq_{bits}bit_{seq_len}seq_{group_size}group"])

            except Exception as e:
                print(f"Error: {e}")
                continue
            del model
            del metrics
            del results
            torch.cuda.empty_cache()
            gc.collect()
shutil.rmtree("./cache", ignore_errors=True)

for b in [4, 8]:
    model, metrics = ControllerQuantizer.gptq(model_name, quant_config={
        "load_in_4bit": b == 4,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "torch.float16",
        "load_in_8bit": b == 8,
        "bnb_8bit_quant_type": "nf8",
        "bnb_8bit_compute_dtype": "torch.float16",
    })
    time_start = time()
    results = evaluator.evaluate(model=model)
    results_obj[f"bnb_{b}bit"] = {
        'base': model_name,
        "eval_time": time()-time_start,
        "ds": "HuggingFaceH4/no_robots test_sft",
        "ds_size": len(evaluator.ds),
        "type": "BNB",
        "method": "N/A",
        "bits": b,
        "model_seqlen": "N/A",
        "q_group_size": "N/A",
        **metrics,
        **results
    }
    with open(results_csv, "a") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results_obj[f"bnb_{b}bit"])
    del model
    del metrics
    torch.cuda.empty_cache()
    gc.collect()
shutil.rmtree("./cache", ignore_errors=True)

print(results_obj)

