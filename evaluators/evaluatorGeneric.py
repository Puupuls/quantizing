from time import time

from transformers import TextGenerationPipeline


class EvaluatorGeneric():
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

    def evaluate(self, model, tokenizer):
        results = self.evaluate_speed(model, tokenizer)
        return results
