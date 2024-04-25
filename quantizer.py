import sys
import json
from time import time
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from transformers import BitsAndBytesConfig, GPTQConfig, AqlmConfig, QuantoConfig, AwqConfig, AutoModelForCausalLM, \
    AutoTokenizer
from utils import get_gpu_utilization


def evaluate_model(model, tokenizer) -> dict[str, dict[str, float]]:
    results = {}

    TaskManager().initialize_tasks()
    results = lm_eval.simple_evaluate(
        model,
        tasks=['glue', 'mmlu', 'wikitext', 'super_glue'],
        num_fewshot=5
    )

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