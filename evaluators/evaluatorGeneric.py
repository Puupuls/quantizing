from time import time

import numpy as np
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from torch.utils._contextlib import F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TextGenerationPipeline, AutoTokenizer


class EvaluatorGeneric:
    def evaluate_speed(self, model, tokenizer):
        result = 0
        try:
            nlp = TextGenerationPipeline(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.8,
            )
            print("Evaluating speed...")
            prompt = 'The great roman empire was '
            tokens_generated = 0
            start = time()

            response = nlp(
                prompt,
            )
            new_text = response[0]['generated_text'].replace(prompt, "")
            tokens_generated += len(tokenizer(new_text)['input_ids'])
            print(f"Generated {prompt} => {new_text}")

            end = time()
            result = tokens_generated / (end - start)
        except Exception as e:
            print(f"Failed to evaluate speed: {e}")
        return result

    def evaluate_similarity(self, model, tokenizer: AutoTokenizer):
        tokenizer.pad_token = tokenizer.eos_token

        models = [
            SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
            SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            SentenceTransformer('headlesstech/semantic_xlmr'),
        ]

        ds = load_dataset('tatsu-lab/alpaca')['train']
        ds = [item for item in ds if len(item['output']) > 0]
        ds = ds[:100]
        similarities = []

        data_loader = DataLoader(ds, batch_size=10, shuffle=False)

        for batch in tqdm(data_loader, desc='Evaluate Similarity'):
            X_templated = []
            Y_encoded = []
            for ins, inp, outp, in zip(batch['instruction'], batch['input'], batch['output']):
                templated = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.'
                if ins:
                    templated += f'\n###Instruction {ins}'
                if inp:
                    templated += f'\n###Input: {inp}'
                templated += f'\n###Response: '
                X_templated.append(templated)
                Y_encoded.append(outp[:100])

            X_encoded = tokenizer(X_templated, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda')
            Y_emb = torch.cat([m.encode(Y_encoded, convert_to_tensor=True) for m in models], dim=1).to('cuda')

            try:
                pred_encodings = []
                Y_pred = model.generate(
                    X_encoded,
                    num_beams=1,
                    max_new_tokens=100,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                decoded = tokenizer.batch_decode(Y_pred, skip_special_tokens=True)
                decoded_diff = [d.replace(x, '') for d, x in zip(decoded, X_templated)]

                # Get next answer
                decoded_end = [d.split('###')[1] if '###' in d else d for d in decoded_diff]
                Y_pred = [d.strip() for d in decoded_end]
                for m in models:
                    pred_encodings.append(m.encode(Y_pred, convert_to_tensor=True))

                sims = torch.nn.functional.cosine_similarity(
                    Y_emb,
                    torch.cat(pred_encodings, dim=1),
                    dim=1
                ).cpu().detach().numpy().tolist()

                similarities.extend(sims)
            except Exception as e:
                print(e)

        return np.mean(similarities)

    def evaluate_perplexity(self, model, tokenizer):

        # Load the tatsu-lab/alpaca dataset
        dataset = load_dataset('tatsu-lab/alpaca')['train']
        dataset = [item['input'] for item in dataset if len(item['input']) > 0]
        dataset = dataset[:100]
        # Tokenize the dataset and create a DataLoader
        tokenizer.pad_token = tokenizer.eos_token
        encoded_inputs = tokenizer(dataset, return_tensors='pt', truncation=True, padding=True, max_length=1024)
        data_loader = DataLoader(encoded_inputs['input_ids'], batch_size=10)

        total_loss = 0
        total_batches = 0

        # Feed the tokenized dataset into the model
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Perplexity"):
                try:
                    outputs = model(input_ids=batch, labels=batch)
                    total_loss += outputs.loss.item()
                    total_batches += len(batch)
                except Exception as e:
                    print(e)

        # Calculate the average loss and convert it to perplexity
        avg_loss = total_loss / (total_batches if total_batches > 0 else 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def calculate_loss(self, model, tokenizer):

        # Load the tatsu-lab/alpaca dataset
        dataset = load_dataset('HuggingFaceH4/no_robots')['test']
        dataset = [item['prompt'] for item in dataset if len(item['prompt']) > 5]
        # Tokenize the dataset and create a DataLoader
        tokenizer.pad_token = tokenizer.eos_token
        encoded_inputs = tokenizer(dataset, return_tensors='pt', truncation=True, padding=True, max_length=1024)
        data_loader = DataLoader(encoded_inputs['input_ids'], batch_size=10)

        total_loss = 0
        total_batches = 0

        # Feed the tokenized dataset into the model
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Loss"):
                try:
                    outputs = model(input_ids=batch, labels=batch)
                    total_loss += outputs.loss.item()
                    total_batches += len(batch)
                except Exception as e:
                    print(e)

        # Calculate the average loss and convert it to perplexity
        avg_loss = total_loss / (total_batches if total_batches > 0 else 1)

        return avg_loss if avg_loss else 1e9
