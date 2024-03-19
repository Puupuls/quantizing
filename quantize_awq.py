from awq import AutoAWQForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

from controllers.ControllerEvaluator import ControllerEvaluator
from datasets import load_dataset

from utils import print_gpu_utilization

model_name = 'mistralai/Mistral-7B-v0.1'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name, cache_dir=f"cache/{model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True, cache_dir=f"cache/{model_name}")
print_gpu_utilization()

model.to('cuda')
print_gpu_utilization()

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(f'outputs/{model_name}/awq/')
tokenizer.save_pretrained(f'outputs/{model_name}/awq/')

model.to('cuda')
print_gpu_utilization()

ds = load_dataset('HuggingFaceH4/no_robots', split='test_sft', cache_dir='cache/HuggingFaceH4/no_robots')
evaluator = ControllerEvaluator()

scores = []
for i in tqdm(ds):
    X = i['messages'][:-1]
    Y = i['messages'][-1]

    X_templated = '\n'.join([f'--> {x["role"]}: {x["content"]}' for x in X])
    X_templated += f'\n--> {Y["role"]}: '

    X_encoded = tokenizer.encode(
        X_templated,
        return_tensors='pt').to('cuda')
    outputs = model.generate(X_encoded, do_sample=True, temperature=0.7, max_new_tokens=1000)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    decoded_x = tokenizer.batch_decode(X_encoded, skip_special_tokens=True)[0]

    decoded_diff = decoded.replace(decoded_x, '')

    # Get next answer
    decoded_end = decoded_diff.split('-->')[1] if '-->' in decoded_diff else decoded_diff
    decoded_answer = decoded_end.strip()

    score = evaluator.evaluate(Y['content'], decoded_answer)
    scores.append(score)

print(f"Mean similarity: {sum(scores) / len(scores)}")
