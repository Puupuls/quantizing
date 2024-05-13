import os
import shutil
import sys
import json
from time import time

import torch
from accelerate import cpu_offload
from awq import AutoAWQForCausalLM
from optimum.gptq import GPTQQuantizer
from tensorboardX import SummaryWriter
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, \
    AutoTokenizer
import pandas as pd

from evaluators.evaluatorGeneric import EvaluatorGeneric
from evaluators.evaluatorSuperGlue import EvaluatorSuperGlue
from utils import get_gpu_utilization
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

CUR_VERSION = 4


def evaluate_model(model, tokenizer, past_result) -> dict[str, dict[str, float]]:
    results = {}

    for i in range(5):
        if past_result['proc_version'] < 3:
            results[f"speed_{i}"] = EvaluatorGeneric().evaluate_speed(model, tokenizer)
            results[f"similarities_{i}"] = EvaluatorGeneric().evaluate_similarity(model, tokenizer)
            results[f"perplexity_{i}"] = EvaluatorGeneric().evaluate_perplexity(model, tokenizer)
            results[f'loss_{i}'] = EvaluatorGeneric().calculate_loss(model, tokenizer)
            results[f'copa_{i}'] = EvaluatorSuperGlue().evaluate_copa(model, tokenizer)
            results[f'boolq_{i}'] = EvaluatorSuperGlue().evaluate_boolq(model, tokenizer)
            results[f'multirc_{i}'] = EvaluatorSuperGlue().evaluate_multirc(model, tokenizer)
        if past_result['proc_version'] < 4:
            results[f'wic_{i}'] = EvaluatorSuperGlue().evaluate_wic(model, tokenizer)
            results[f'axg_{i}'] = EvaluatorSuperGlue().evaluate_axg(model, tokenizer)

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
        time_start = time()
        params = {k: v for k, v in params.items() if k not in ['model_name']}
        quantizer = GPTQQuantizer(
            **params,
            pad_token_id=tokenizer.eos_token_id,
        )
        model = quantizer.quantize_model(model, tokenizer)
        model = cpu_offload(model, offload_buffers=True)
        metrics["quantization_time"] = time() - time_start
        metrics["gpu_utilization"] = get_gpu_utilization()

    elif quantization_method == 'bitsandbytes':
        config = BitsAndBytesConfig(
            **params,
            load_in_4bit=params['bits'] == 4,
            load_in_8bit=params['bits'] == 8,
        )
        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(
            params['model_name'],
            device_map="auto",
            quantization_config=config
        )
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

    tokenizer.pad_token = tokenizer.eos_token
    results = evaluate_model(model, tokenizer, past_results)
    results.update(metrics)
    return results


def main(args=None):
    if args is None:
        args = sys.argv
    # Parse the command-line arguments.
    quantization_method = args[1]
    run_name = args[2]
    print(args)
    params = json.loads(' '.join(args[3:]))

    results_csv = f"results/{run_name}.csv"
    # read existing csv
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Check if results already computed
    filtered_df = pd.DataFrame()
    if not df.empty:
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
        json.dump(prev_result, sys.stderr)
        return prev_result

    # Run the quantization.
    res = run_quantization(quantization_method, params, prev_result)
    result = {
        **params,
        'method': quantization_method,
        **prev_result,
        **res,
        'proc_version': CUR_VERSION,
    }

    # Save the results to a CSV file.
    if filtered_df.empty:
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = df.drop(filtered_df.index)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    os.makedirs('results', exist_ok=True)
    df.to_csv(results_csv, index=False)

    json.dump(result, sys.stderr)
    return result


if __name__ == '__main__':
    main()
