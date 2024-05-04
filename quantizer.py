import os
import shutil
import sys
import json
from time import time

import torch
from awq import AutoAWQForCausalLM
from optimum.gptq import GPTQQuantizer
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, \
    AutoTokenizer
import pandas as pd

from evaluators.evaluatorGeneric import EvaluatorGeneric
from evaluators.evaluatorSuperGlue import EvaluatorSuperGlue
from utils import get_gpu_utilization
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


def evaluate_model(model, tokenizer) -> dict[str, dict[str, float]]:
    results = {}
    results.update(EvaluatorGeneric().evaluate(model, tokenizer))
    results.update(EvaluatorSuperGlue().evaluate(model, tokenizer))

    return results

def run_quantization(quantization_method: str, params: dict[str, any]) -> dict:
    metrics = {
        "gpu_background_ram": get_gpu_utilization(),
        "gpu_utilization": 0,
        "quantization_time": 0,
        "evaluation_time": 0,
    }
    if quantization_method == 'gptq':
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], torch_dtype=torch.float16, device_map="auto")
        time_start = time()
        params = {k: v for k, v in params.items() if k not in ['model_name']}
        quantizer = GPTQQuantizer(**params)
        model = quantizer.quantize_model(model, tokenizer)
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()

    elif quantization_method == 'bitsandbytes':
        config = BitsAndBytesConfig(
            **params
        )
        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto", quantization_config=config)
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'], trust_remote_code=True)
    elif quantization_method == 'awq':
        model = AutoAWQForCausalLM.from_pretrained(params['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'], trust_remote_code=True)
        time_start = time()
        params = {k: v for k, v in params.items() if k not in ['model_name']}
        model.quantize(tokenizer, quant_config=params)
        metrics["quantization_time"] = time() - time_start

        # Save and load with fused layers
        model.save_quantized(f"cache_awq")
        del model
        # model = AutoAWQForCausalLM.from_quantized(f"cache_awq", fuse_layers=True)
        model = AutoModelForCausalLM.from_pretrained(f"cache_awq", device_map="auto")
        shutil.rmtree(f"cache_awq", ignore_errors=True)
        metrics["gpu_utilization"] = get_gpu_utilization()
    else:
        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto")
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'], trust_remote_code=True)

    time_start = time()
    results = evaluate_model(model, tokenizer)
    metrics["evaluation_time"] = time() - time_start
    results.update(metrics)
    return results

def main():
    # Parse the command-line arguments.
    quantization_method = sys.argv[1]
    run_name = sys.argv[2]
    params = json.loads(sys.argv[3])

    results_csv = f"results/{run_name}.csv"
    # read existing csv
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Check if results already computed
    filtered_df = df[(df['model_name'] == params['model_name']) & (df['method'] == quantization_method)]
    if not filtered_df.empty:
        for key, value in params.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
    if not filtered_df.empty:
        print(f"Skipping {quantization_method} for {params['model_name']}")
        return

    # Run the quantization.
    result = {
        **params,
        'method': quantization_method,
        **run_quantization(quantization_method, params)
    }

    # Save the results to a CSV file.
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    os.makedirs('results', exist_ok=True)
    df.to_csv(results_csv, index=False)


if __name__ == '__main__':
    main()