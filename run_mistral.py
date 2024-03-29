import gc
from time import time

import torch
from awq import AutoAWQForCausalLM
from csv import DictWriter
from transformers import AutoTokenizer

from controllers.ConrollerQuantizer import ControllerQuantizer
from controllers.ControllerEvaluator import ControllerEvaluator
from utils import get_gpu_utilization

gpu_background_ram = get_gpu_utilization()
model_name = "mistralai/Mistral-7B-v0.1"

load_start = time()
model = AutoAWQForCausalLM.from_pretrained(model_name, cache_dir=f"cache/").to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=f"cache/")
load_end = time()
gpu_loaded = get_gpu_utilization()

results_csv = model_name.replace("/",'_')+"--awq.csv"
results_obj = {}

time_start = time()
evaluator = ControllerEvaluator(tokenizer=tokenizer)

time_start = time()
result = evaluator.evaluate(model=model)
results_obj["original"] = {
    "eval_time": time()-time_start,
    "ds": "HuggingFaceH4/no_robots test_sft",
    "ds_size": len(evaluator.ds),
    "similarity": result,
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
}
with open(results_csv, "w") as f:
    writer = DictWriter(f, fieldnames=list(results_obj['original'].keys()))
    writer.writeheader()
    writer.writerow(results_obj["original"])

del model
del result
torch.cuda.empty_cache()
gc.collect()


# for i in [4]:  # only 4bit supported for now
#     for m in ["GEMM"]:  # GEMV is crashing
model, metrics = ControllerQuantizer.awq(model_name, quant_config={"version": "GEMM", "w_bit": 4})
time_start = time()
result = evaluator.evaluate(model=model)
results_obj[f"awq"] = {
    "eval_time": time()-time_start,
    "ds": "HuggingFaceH4/no_robots test_sft",
    "ds_size": len(evaluator.ds),
    "similarity": result,
    "type": "AWQ",
    "method": "GEMM",
    "bits": 4,
    "model_seqlen": "N/A",
    "q_group_size": "N/A",
    **metrics
}
with open(results_csv, "a") as f:
    writer = DictWriter(f, fieldnames=list(results_obj['original'].keys()))
    writer.writerow(results_obj[f"awq"])

del model
del metrics
del result
torch.cuda.empty_cache()
gc.collect()

for bits in range(2, 8):
    for seq_len in [512, 1024, 2048]:
        for group_size in [128, 256, 512]:
            try:
                model, metrics = ControllerQuantizer.gptq(model_name, quant_config={
                    "bits": bits,
                    "model_seqlen": seq_len,
                    "q_group_size": group_size,
                })
                time_start = time()
                result = evaluator.evaluate(model=model)
                results_obj[f"gptq_{bits}bit_{seq_len}seq_{group_size}group"] = {
                    "eval_time": time()-time_start,
                    "ds": "HuggingFaceH4/no_robots test_sft",
                    "ds_size": len(evaluator.ds),
                    "similarity": result,
                    "type": "GPTQ",
                    "method": "N/A",
                    "bits": bits,
                    "model_seqlen": seq_len,
                    "q_group_size": group_size,
                    **metrics
                }
                with open(results_csv, "a") as f:
                    writer = DictWriter(f, fieldnames=list(results_obj['original'].keys()))
                    writer.writerow(results_obj[f"gptq_{bits}bit_{seq_len}seq_{group_size}group"])

                del model
                del metrics
                del result
            except Exception as e:
                print(f"Error: {e}")
                continue
            torch.cuda.empty_cache()
            gc.collect()

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
    result = evaluator.evaluate(model=model)
    results_obj[f"bnb_{b}bit"] = {
        "eval_time": time()-time_start,
        "ds": "HuggingFaceH4/no_robots test_sft",
        "ds_size": len(evaluator.ds),
        "similarity": result,
        "type": "BNB",
        "method": "N/A",
        "bits": b,
        "model_seqlen": "N/A",
        "q_group_size": "N/A",
        **metrics
    }
    with open(results_csv, "a") as f:
        writer = DictWriter(f, fieldnames=list(results_obj['original'].keys()))
        writer.writerow(results_obj[f"bnb_{b}bit"])

    del model
    del metrics
    del result
    torch.cuda.empty_cache()
    gc.collect()

print(results_obj)

