import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class ControllerEvaluator:
    def __init__(self):
        self.models = [
            SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
            SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            SentenceTransformer('headlesstech/semantic_xlmr'),
        ]

    def evaluate(self, Y_true, Y_pred):
        true_encodings = []
        pred_encodings = []

        for model in self.models:
            true_encodings.append(model.encode(Y_true))
            pred_encodings.append(model.encode(Y_pred))

        true_encoding = np.concatenate(true_encodings)
        pred_encoding = np.concatenate(pred_encodings)

        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(true_encoding),
            torch.tensor(pred_encoding),
            dim=0
        ).item()

        return similarity
