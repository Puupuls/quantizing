import subprocess
import json

from tqdm.contrib import itertools

# Define the parameter space for each quantization method.
MODEL = ['mistralai/Mistral-7B-v0.1', 'Undi95/Meta-Llama-3-8B-hf']
RUN_NAME = 'run_0'

awq_params = {
    'model_name': MODEL,
    'w_bit': [4],
    'version': ['GEMM', 'GEMV'],
    'zero_point': [True],
    'q_group_size': [64, 128, 256, 512],
}

gptq_params = {
    'model_name': MODEL,
    'bits': [2, 3, 4, 8],
    'disable_exllama': [True, False],
    'model_seqlen': [512, 1024, 2048],
    'group_size': [64, 128, 256, 512, -1],
    'dataset': ['c4-new', 'wikitext2'],
    'damp_percent': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'desc_act': [True, False],
    'sym': [True, False],
    'true_sequential': [True, False],
    'batch_size': [1, 2, 4, 8],
}

bits_and_bytes_params = {
    'model_name': MODEL,
    'bits': [4, 8],
    'llm_int8_threshold': [6, 7, 5, 4],
    'bnb_4bit_quant_type': ['fp4', 'nf4'],
    'bnb_4bit_use_double_quant': [True, False],
}

# Combine all parameter spaces into a single dictionary.
param_spaces = {
    'original': {
        'model_name': MODEL,
    },
    'awq': awq_params,
    'gptq': gptq_params,
    'bitsandbqytes': bits_and_bytes_params,
}

failures = []
# Iterate over all quantization methods.
for method, param_space in param_spaces.items():
    # Generate all combinations of parameters for this method.
    param_names = param_space.keys()
    param_values = param_space.values()
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        if method == 'gptq' and params['bits'] != 8 and params['disable_exllama']:
            continue

        # Prepare the command to run the quantization in a separate process.
        try:
            command = ['python', 'quantizer.py', method, RUN_NAME, json.dumps(params)]
            print(f'Running command: {" ".join(command)}')
            # Run the command and wait for it to finish.
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            failures.append({
                'method': method,
                'params': params,
            })
            with open('failures.json', 'w') as f:
                json.dump(failures, f)