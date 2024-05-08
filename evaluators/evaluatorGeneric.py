from time import time

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from torch.utils._contextlib import F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TextGenerationPipeline


class EvaluatorGeneric:
    def evaluate_speed(self, model, tokenizer):
        results = {}

        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=20)
        prompt = 'Hello! '
        tokens_generated = 0
        start = time()
        for i in range(5):
            response = nlp(prompt)
            new_text = response[0]['generated_text'].replace(prompt, "")
            tokens_generated += len(tokenizer(new_text)['input_ids'])
        end = time()
        results["speed_s_tokens"] = tokens_generated
        results["speed_s_speed"] = tokens_generated / (end - start)
        print(f"Short test took {end - start} seconds at {results['speed_s_speed']} tokens per second")

        prompt = 'Write 3000 word essay on the topic of ancient Rome: '
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=500)
        tokens_generated = 0
        start = time()
        for i in range(5):
            response = nlp(prompt)
            new_text = response[0]['generated_text'].replace(prompt, "")
            tokens_generated += len(tokenizer(new_text)['input_ids'])
        end = time()
        results["speed_l_tokens"] = tokens_generated
        results["speed_l_speed"] = tokens_generated / (end - start)
        print(f"Long test took {end - start} seconds at {results['speed_l_speed']} tokens per second")

        return results


    def evaluate_bleu(self, model, tokenizer):
        results = {}

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['validation']
        dataset = [item['text'] for item in dataset]
        prompts_and_references = []
        for line in dataset:
            # Use the first sentence as the prompt and the rest as the reference text
            sentences = line.split('. ')
            if len(sentences) > 1:
                prompt = sentences[0]
                reference_text = ' '.join(sentences[1:])
                prompts_and_references.append((prompt, reference_text))
            if len(prompts_and_references) >= 1000:
                break

        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=250)
        scores = []
        start = time()
        batches = [prompts_and_references[i:i + 25] for i in range(0, len(prompts_and_references), 25)]

        for batch in tqdm(batches, desc="BLEU score"):
            batch_prompts, batch_references = zip(*batch)
            batch_responses = nlp(list(batch_prompts))

            for response, reference_text in zip(batch_responses, batch_references):
                generated_text = response['generated_text'].split('.', 1)[-1].strip()  # get generated text
                candidate_text = generated_text.split()
                reference_text = reference_text.split()
                score = sentence_bleu([reference_text], candidate_text)
                scores.append(score)

        results["bleu_score"] = sum(scores) / len(scores)
        results["bleu_time"] = time() - start
        print(f"Average BLEU score: {results['bleu_score']}")

        return results

    def evaluate_perplexity(self, model, tokenizer):
        results = {}

        # Load the wikitext dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['validation']
        dataset = [item['text'] for item in dataset if len(item['text']) > 0]
        dataset = dataset[:1000]
        start = time()
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
                outputs = model(input_ids=batch, labels=batch)
                total_loss += outputs.loss.item()
                total_batches += len(batch)

        # Calculate the average loss and convert it to perplexity
        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        results["perplexity"] = perplexity
        results["perplexity_time"] = time() - start

        return results


    def evaluate(self, model, tokenizer):
        results = {}
        results.update(self.evaluate_speed(model, tokenizer))
        results.update(self.evaluate_perplexity(model, tokenizer))
        return results
