import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class ControllerEvaluator:
    def __init__(self, tokenizer):
        self.models = [
            SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
            SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            SentenceTransformer('headlesstech/semantic_xlmr'),
        ]
        self.ds = load_dataset('HuggingFaceH4/no_robots', split='test', cache_dir='cache/HuggingFaceH4/no_robots')
        self.tokenizer = tokenizer

        self.ds_ready = self.prepare_ds()

    def evaluate(self, model, generate_config=None):
        _generate_config = {
            "num_beams": 1,
            "max_new_tokens": 200,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "use_cache": True,
        }
        failures = 0
        if generate_config:
            _generate_config.update(generate_config)

        similarities = []
        bleus = []
        for i, d in tqdm(enumerate(self.ds_ready), desc='Evaluating', total=len(self.ds_ready)):
            try:
                X = d['X']
                Y_emb = d['Y_embedded']
                X_decoded = d['X_decoded']
                pred_encodings = []
                Y_pred = model.generate(X[:250], **_generate_config, pad_token_id=self.tokenizer.eos_token_id)
                Y_pred = self.extract_y(X_decoded, Y_pred)
                for m in self.models:
                    pred_encodings.append(m.encode(Y_pred, convert_to_tensor=True))

                similarities.append(
                    torch.nn.functional.cosine_similarity(
                        Y_emb,
                        torch.concatenate(pred_encodings),
                        dim=0
                    ).item()
                )
                bleus.append(sentence_bleu(
                    [d['Y_decoded']],
                    [Y_pred],
                    smoothing_function=SmoothingFunction().method1
                ))
            except Exception as e:
                print(f"Error at {i}: {e}")
                failures += 1
        return {
            "similarities": np.mean(similarities),
            "bleus": np.mean(bleus),
            "failures": failures
        }

    def prepare_ds(self):
        ds_ready = []
        for i in tqdm(self.ds, desc='Preparing dataset'):
            X = i['messages'][:-1]
            Y = i['messages'][-1]

            X_templated = '\n'.join([f'### {x["role"]}: {x["content"]}' for x in X])
            X_templated += f'\n### {Y["role"]}: '

            X_encoded = self.tokenizer.encode(
                X_templated,
                return_tensors='pt').to('cuda')
            Y_encoded = self.tokenizer.encode(
                Y['content'],
                return_tensors='pt').to('cuda')
            ds_ready.append({
                "X": X_encoded,
                "Y": Y_encoded,
                "Y_embedded":
                torch.concatenate([m.encode(Y['content'], convert_to_tensor=True) for m in self.models]),
                "X_decoded": self.tokenizer.batch_decode(X_encoded, skip_special_tokens=True)[0],
                "Y_decoded": Y['content'],
                # Need to decode in case tokenizer removes some characters
            })

        return ds_ready

    def extract_y(self, X_decoded, pred):
        decoded = self.tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        decoded_diff = decoded.replace(X_decoded, '')

        # Get next answer
        decoded_end = decoded_diff.split('###')[1] if '###' in decoded_diff else decoded_diff
        decoded_answer = decoded_end.strip()
        return decoded_answer
