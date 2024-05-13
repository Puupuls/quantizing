import random
from time import time

from datasets import load_dataset
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import pipeline, TextGenerationPipeline
MAX_PER_TASK = 100

class EvaluatorSuperGlue():
    def evaluate_axb(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'axb')
        dataset = [i for i in dataset['test']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating axb B1', total=len(dataset)):
            # Combine 'sentence1' and 'sentence2' into a single string
            input_data = "ALWAYS respond with one and only one symbol 0=yes, 1=no, nothing else, no punctuation! The task is to determine whether a 2nd sentence is entailed from the 1st sentence or not.: " + \
                         example['sentence1'] + ' ' + example['sentence2'] + "\nAnswer: "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_axb_accuracy_b1'] = accuracy
        results['sg_axb_time_b1'] = (time() - time_start) / total

        # now in batches of 8
        correct = 0
        total = 0
        time_start = time()
        for i in tqdm(range(0, len(dataset), 8), desc='Evaluating axb B8', total=len(dataset) // 8):
            # Combine 'sentence1' and 'sentence2' into a single string for each example in the batch
            input_data_batch = []
            for j in range(8):
                example = dataset[i + j]
                input_data_batch.append("ALWAYS respond with one and only one symbol 0=yes, 1=no, nothing else, no punctuation! The task is to determine whether a 2nd sentence is entailed from the 1st sentence or not.: " + \
                                        example['sentence1'] + ' ' + example['sentence2'] + "\nAnswer: ")
            predictions = nlp(
                input_data_batch,
            )
            for j in range(8):
                prediction = predictions[j]
                example = dataset[i + j]
                answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
                if answer == str(example['label']):
                    correct += 1
                total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_axb_accuracy_b8'] = accuracy
        results['sg_axb_time_b8'] = (time() - time_start) / total
        results['sg_axb_samples'] = total

        return results

    def evaluate_axg(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'axg')
        dataset = [i for i in dataset['test']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        # Evaluate
        for example in tqdm(dataset, desc='Evaluating axg', total=len(dataset)):
            # Combine 'sentence1' and 'sentence2' into a single string
            input_data = ("ALWAYS respond with one and only one symbol 0=yes, 1=no, nothing else, no punctuation! "
                          "The task is to determine whether a 2nd sentence is entailed from the 1st sentence or not. "
                          f"Sentence 1: {example['hypothesis']}"
                          f"\nSentence 2: {example['premise']}"
                          "\nAnswer: "
                          )
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total

        return accuracy

    def evaluate_boolq(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'boolq')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating boolq B1', total=len(dataset)):
            # Combine 'passage' and 'question' into a single string and add a prompt for a boolean answer
            input_data = example['passage'] + ' ' + example['question'] + "\nAnswer: 1 for True, 0 for False.\nAnswer: "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_boolq_accuracy_b1'] = accuracy
        results['sg_boolq_time_b1'] = (time() - time_start) / total
        results['sg_boolq_samples'] = total

        return accuracy

    def evaluate_cb(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'cb')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating cb', total=len(dataset)):
            # Combine 'premise' and 'hypothesis' into a single string
            input_data = "ALWAYS respond with one and only one symbol 0=no, 1=yes, nothing else, no punctuation! The task is to determine whether the hypothesis is entailed from the premise: " + \
                         example['premise'] + '\n Hypothesis: ' + example['hypothesis'] + "\nAnswer: "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_cb_accuracy'] = accuracy
        results['sg_cb_time'] = (time() - time_start) / total
        results['sg_cb_samples'] = total

        return results

    def evaluate_copa(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'copa')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating copa', total=len(dataset)):
            # Combine 'premise' and 'choice1' and 'choice2' into a single string
            input_data = "ALWAYS respond with one and only one symbol 0=choice1, 1=choice2, nothing else, no punctuation! The task is to determine the more plausible cause or effect of the premise: " + \
                         example['premise'] + '\nChoice1: ' + example['choice1'] + '\nChoice2: ' + example[
                             'choice2'] + "\nAnswer: "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_copa_accuracy'] = accuracy
        results['sg_copa_time'] = (time() - time_start) / total
        results['sg_copa_samples'] = total

        return accuracy

    def evaluate_multirc(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'multirc')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating multirc', total=len(dataset)):
            # Combine 'paragraph', 'question', and 'answer' into a single string
            input_data = "ALWAYS respond with one and only one symbol 0=false, 1=true, nothing else, no punctuation! The task is to determine if the answer to the question about the paragraph is true or not: " + \
                         example['paragraph'] + '\nQuestion: ' + example['question'] + '\nAnswer: ' + example['answer'] + "\nIs the answer true? "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_multirc_accuracy'] = accuracy
        results['sg_multirc_time'] = (time() - time_start) / total
        results['sg_multirc_samples'] = total

        return accuracy

    def evaluate_record(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'record')
        dataset = [i for i in dataset['validation']]
        dataset = dataset[:MAX_PER_TASK]

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=20)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating record', total=len(dataset)):
            # Combine 'passage', 'question', and 'entities' into a single string
            input_data = "ALWAYS respond with one and only one entity name from the passage that answers the question. End your answer with newline character. The task is to answer the question based on the passage: " + \
                         example['passage'] + '\nQuestion: ' + example['query'] + '\nEntities: ' + ', '.join(
                example['entities']) + "\nAnswer: "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!').replace(input_data, '').split('\n')[0]
            for a in example['answers']:
                if a in answer:
                    correct += 1
                    break
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_record_accuracy'] = accuracy
        results['sg_record_time'] = (time() - time_start) / total
        results['sg_record_samples'] = total

        return results

    def evaluate_rte(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'rte')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating rte', total=len(dataset)):
            # Combine 'sentence1' and 'sentence2' into a single string
            input_data = "ALWAYS respond with one and only one symbol 1=no, 0=yes, nothing else, no punctuation! The task is to determine whether the hypothesis is an entailment of the premise: " + \
                         example['premise'] + '\nHypothesis: ' + example['hypothesis'] + "\nIs the hypothesis an entailment of the premise? "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_rte_accuracy'] = accuracy
        results['sg_rte_time'] = (time() - time_start) / total
        results['sg_rte_samples'] = total

        return results

    def evaluate_wic(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'wic')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating wic', total=len(dataset)):
            # Combine 'sentence1', 'sentence2' and 'word' into a single string
            input_data = ("ALWAYS respond with one and only one symbol 0=no, 1=yes, nothing else, no punctuation! "
                          "The task is to determine whether the target word has the same meaning in both sentences: "
                          f"\nSentence 1: {example['sentence1']}"
                          f"\nSentence 2: {example['sentence2']}"
                          f"\nWord: {example['word']}"
                          "\nAnswer: ")
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total

        return accuracy

    def evaluate_wsc(self, model, tokenizer):
        # Load the dataset
        dataset = load_dataset('super_glue', 'wsc.fixed')
        dataset = [i for i in dataset['validation']]
        dataset_1 = [i for i in dataset if i['label'] == 1]
        dataset_0 = [i for i in dataset if i['label'] == 0]
        dataset = dataset_1[:MAX_PER_TASK//2] + dataset_0[:MAX_PER_TASK//2]
        random.shuffle(dataset)

        # Create a pipeline
        nlp = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=1)

        # Initialize metrics
        correct = 0
        total = 0
        time_start = time()

        results = {}

        # Evaluate
        for example in tqdm(dataset, desc='Evaluating wsc', total=len(dataset)):
            # Combine 'text', 'span1_text', and 'span2_text' into a single string
            input_data = "ALWAYS respond with one and only one symbol 0=no, 1=yes, nothing else, no punctuation! The task is to determine whether the pronoun refers to the named entity: " + \
                         example['text'] + '\nEntity: ' + example['span1_text'] + '\nPronoun: ' + example['span2_text'] + "\nDoes the pronoun refer to the named entity? "
            prediction = nlp(
                input_data,
            )
            answer = prediction[0]['generated_text'].strip('\n \t.,!')[-1]
            if answer == str(example['label']):
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total
        results['sg_wsc_accuracy'] = accuracy
        results['sg_wsc_time'] = (time() - time_start) / total
        results['sg_wsc_samples'] = total

        return results