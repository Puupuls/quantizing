import os
import subprocess
import json

import torch
from hebo.design_space import DesignSpace
from ray import tune, air
from ray.tune.search.hebo import HEBOSearch

# Define the parameter space for each quantization method.
MODEL = ['mistralai/Mistral-7B-v0.1', 'Undi95/Meta-Llama-3-8B-hf', 'meta-llama/Llama-2-7b-hf']
RUN_NAME = 'run_3'

awq_params = {
    'model_name': tune.choice(MODEL),
    'w_bit': tune.choice([4]),
    'version': tune.choice(['GEMM', 'GEMV']),
    'zero_point': tune.choice([True]),
    'q_group_size': tune.choice([64, 128, 256, 512]),
    'method': tune.choice(['awq']),
}

gptq_params = {
    'model_name': tune.choice(MODEL),
    'bits': tune.choice([2, 3, 4, 8]),
    'model_seqlen': tune.choice([512, 1024, 2048]),
    'exllama_version': tune.choice([1, 2]),
    'group_size': tune.choice([64, 128, 256, 512]),
    'dataset': tune.choice(['c4-new', 'wikitext2']),
    'damp_percent': tune.choice([0.1, 0.2, 0.5, 0.75]),
    'desc_act': tune.choice([True, False]),
    'sym': tune.choice([True, False]),
    'true_sequential': tune.choice([True, False]),
    'batch_size': tune.choice([1, 4, 8]),
    'method': tune.choice(['gptq']),
}

bits_and_bytes_params = {
    'model_name': tune.choice(MODEL),
    'llm_int8_threshold': tune.choice([6., 7., 5., 4.]),
    # 'bnb_4bit_quant_type': tune.choice(['fp4', 'nf4']),
    'bnb_4bit_use_double_quant': tune.choice([True, False]),
    'method': tune.choice(['bitsandbytes']),
    'block_size': tune.choice([64, 256, 512, 2048]),
    'bits': tune.choice([4, 8]),
}

# Combine all parameter spaces into a single dictionary.
param_spaces = [{
        'model_name': tune.choice(MODEL + ['TheBloke/Mistral-7B-v0.1-AWQ', 'TheBloke/Mistral-7B-v0.1-GPTQ', 'TheBloke/Llama-2-7B-AWQ', 'TheBloke/Llama-2-7B-GPTQ']),
        'method': tune.choice(['original']),
    },
    awq_params,
    gptq_params,
    bits_and_bytes_params,
]

def quantization_process(config):
    method = config['method']
    params = {k: v for k, v in config.items() if k != 'method'}
    command = ['python', 'quantizer.py', method, RUN_NAME, f"{json.dumps(params)}"]
    print(f'Running command: {" ".join(command)}')
    if method == 'gptq':
        if params['exllama_version'] == 2 and params['bits'] != 4:
            raise ValueError('ExLLAMA version 2 only supports 4-bit quantization.')
    # Run the command and wait for it to finish.
    # result = subprocess.run(command, check=True, capture_output=True)
    import quantizer as q
    res = q.main(command[1:])
    # print(result.stderr)

    # res = json.loads(result.stderr)

    # Parse the output to get the objective value.
    losses = [res[f'loss_{i}'] for i in range(10) if f'loss_{i}' in res.keys() and res[f'loss_{i}'] is not None]
    objective_value = sum(losses) / len(losses)
    return {
        'score': objective_value,
        **res,
    }


num_cpus = os.cpu_count()
print(f'Number of CPUs: {num_cpus}')
num_gpus = torch.cuda.device_count()
print(f'Number of GPUs: {num_gpus}')

for space in param_spaces:

    # Define the experiment configuration.
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'

    space_size = 1
    for k, v in space.items():
        space_size *= len(v)

    try:
        analysis = tune.run(
            quantization_process,
            config=space,
            search_alg=HEBOSearch(metric='score', mode='min'),
            resources_per_trial={
                'cpu': 1,
                'gpu': 1,
            },
            name=f'{space["method"][0]}_quantization',
            num_samples=30 if space_size > 30 else space_size,
            verbose=0,
        )
    except Exception as e:
        print(f'Failed to run experiment: {e}')
        continue
