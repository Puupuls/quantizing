import os
import shutil
import sys
import json
from time import time

import torch
from accelerate import cpu_offload
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

CUR_VERSION = 2


def evaluate_model(model, tokenizer, past_result) -> dict[str, dict[str, float]]:
    results = {}

    if past_result['proc_version'] < 1:
        results.update(EvaluatorGeneric().evaluate(model, tokenizer))
        results.update(EvaluatorSuperGlue().evaluate(model, tokenizer))
    elif past_result['proc_version'] < 2:
        results.update(EvaluatorGeneric().evaluate_perplexity(model, tokenizer))

    return results


def run_quantization(quantization_method: str, params: dict[str, any], past_results) -> dict:
    metrics = {
        "gpu_background_ram": get_gpu_utilization(),
        "gpu_utilization": 0,
        "quantization_time": 0,
    }
    if quantization_method == 'gptq':
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], torch_dtype=torch.float16, device_map="auto")
        model = cpu_offload(model, offload_buffers=True)
        time_start = time()
        params = {k: v for k, v in params.items() if k not in ['model_name']}
        quantizer = GPTQQuantizer(
            **params,
            pad_token_id=tokenizer.eos_token_id,
        )
        model = quantizer.quantize_model(model, tokenizer)
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()

    elif quantization_method == 'bitsandbytes':
        config = BitsAndBytesConfig(
            **params
        )
        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto", quantization_config=config)
        model = cpu_offload(model, offload_buffers=True)
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
        model = cpu_offload(model, offload_buffers=True)
        shutil.rmtree(f"cache_awq", ignore_errors=True)
        metrics["gpu_utilization"] = get_gpu_utilization()
    else:
        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto", torch_dtype=torch.float16)
        model = cpu_offload(model, offload_buffers=True)
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'], trust_remote_code=True)

    results = evaluate_model(model, tokenizer, past_results)
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

    prev_result = {
        'proc_version': 0
    }
    if not filtered_df.empty:
        prev_result = filtered_df.to_dict(
            orient='records',
        )[0]
        # replace nan with null
        for key, value in prev_result.items():
            if pd.isna(value):
                prev_result[key] = None
        if 'proc_version' not in prev_result or prev_result['proc_version'] is None:
            prev_result['proc_version'] = 1
        print(f"Found previous results for {params['model_name']} with {quantization_method} quantization. Version {prev_result['proc_version']}.")

    if prev_result['proc_version'] == CUR_VERSION:
        print(f"Already processed {params['model_name']} with {quantization_method} quantization. Version {prev_result['proc_version']}.")
        return

    # Run the quantization.
    result = {
        **params,
        'method': quantization_method,
        **prev_result,
        **run_quantization(quantization_method, params, prev_result),
        'proc_version': '2'
    }

    # Save the results to a CSV file.
    if filtered_df.empty:
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = df.drop(filtered_df.index)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    os.makedirs('results', exist_ok=True)
    df.to_csv(results_csv, index=False)


if __name__ == '__main__':
    main()
