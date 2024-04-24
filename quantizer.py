import sys
import json
from time import time

import evaluate
import torch
from datasets import load_metric, load_dataset
from evaluate import evaluator
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig, GPTQConfig, AqlmConfig, QuantoConfig, AwqConfig, AutoModelForCausalLM, \
    AutoTokenizer
from utils import get_gpu_utilization


models = [
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    SentenceTransformer('headlesstech/semantic_xlmr'),
]
def similarity(predictions, references):
    y_emb = []
    y_pred_emb = []
    for i, d in enumerate(predictions):
        pass


def evaluate_model(model, tokenizer) -> dict[str, dict[str, float]]:
    # Define the hardcoded datasets and metrics
    # datasets = {
    #     'HuggingFaceH4/no_robots': ['bleu', 'rouge', 'perplexity', 'accuracy', 'f1' 'precision'],
    #     'cais/mmlu': ['bleu', 'rouge', 'perplexity', 'accuracy', 'f1' 'precision'],
    #     'tau/commonsense_qa': ['bleu', 'rouge', 'perplexity', 'accuracy', 'f1' 'precision'],
    #     'openai_humaneval': ['bleu', 'rouge', 'perplexity', 'accuracy', 'f1' 'precision'],
    #     'quac': ['bleu', 'rouge', 'perplexity', 'accuracy', 'f1' 'precision'],
    #     'rajpurkar/squad_v2': ['squad_v2'],
    #     'super_glue': ['super_glue'],
    #     'wiki_split': ['wiki_split'],
    # }
    results = {}

    task_evaluator = evaluator("question-answering")
    data = load_dataset("squad_v2", split="validation[:1000]")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=data,
        tokenizer=tokenizer,
        metric="squad",
        strategy="bootstrap",
        n_resamples=10,
    )
    results[f"squad_total_time_in_sec"] = eval_results["total_time_in_seconds"]
    results[f"squad_samples_per_second"] = eval_results["samples_per_second"]
    results[f"squad_latency_sec"] = eval_results["latency_in_seconds"]
    results[f"squad_exact_match_score"] = eval_results["exact_match"]['score']
    results[f"squad_exact_match_error"] = eval_results["exact_match"]['standard_error']
    results[f"squad_exact_match_confidence_min"] = eval_results["exact_match"]['confidence_interval'][0]
    results[f"squad_exact_match_confidence_max"] = eval_results["exact_match"]['confidence_interval'][1]
    results[f"squad_f1_score"] = eval_results["f1"]['score']
    results[f"squad_f1_error"] = eval_results["f1"]['standard_error']
    results[f"squad_f1_confidence_min"] = eval_results["f1"]['confidence_interval'][0]
    results[f"squad_f1_confidence_max"] = eval_results["f1"]['confidence_interval'][1]

    return results

def run_quantization(quantization_method: str, params: dict[str, any]) -> dict:
    metrics = {
        "gpu_background_ram": get_gpu_utilization(),
        "gpu_utilization": 0,
        "quantization_time": 0,
        "evaluation_time": 0,
    }
    config = None
    if quantization_method == 'gptq':
        config = GPTQConfig(
            **params
        )
    elif quantization_method == 'bitsandbytes':
        config = BitsAndBytesConfig(
            **params
        )
    elif quantization_method == 'awq':
        config = AwqConfig(
            **params
        )
    elif quantization_method == 'aqlm':
        config = AqlmConfig(
            **params
        )
    elif quantization_method == 'quanto':
        config = QuantoConfig(
            **params
        )

    time_start = time()
    model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto", quantization_config=config)
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

    # Run the quantization.
    result = run_quantization(quantization_method, params)

    # Append the result to the results file. Creating it if not exists
    with open(f"{run_name}.csv", "a") as f:
        f.write(result)


if __name__ == '__main__':
    main()