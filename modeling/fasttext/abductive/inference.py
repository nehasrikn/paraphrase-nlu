from modeling.fasttext.lexical_classifier import FastTextClassifier
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset, ParaphrasedAbductiveNLIExample, anli_dataset
from typing import Union, List, Any, Dict
from dataclasses import asdict
from annotated_data.annotated_data import anli_human
from utils import PROJECT_ROOT_DIR, write_json, load_json
import os
from tqdm import tqdm
import string
import random


ANLI = "__label__{label} {obs1} {hyp1} {obs1} [SEP] {obs1} {hyp2} {obs1}"
ANLI_PARTIAL_INPUT = "__label__{label} {hyp1} {hyp2}"

def format_example(e: Union[AbductiveNLIExample, ParaphrasedAbductiveNLIExample], partial_input=False):
    e_inputs = {
        'obs1': e.obs1 if isinstance(e, AbductiveNLIExample) else e.original_example.obs1,
        'obs2' : e.obs2 if isinstance(e, AbductiveNLIExample) else e.original_example.obs2,
        'hyp1': e.hyp1 if isinstance(e, AbductiveNLIExample) else e.original_example.hyp1,
        'hyp2': e.hyp2 if isinstance(e, AbductiveNLIExample) else e.original_example.hyp2,
        'label': e.modeling_label if isinstance(e, AbductiveNLIExample) else e.original_example.modeling_label,
    }
    for k in e_inputs:
        if k == 'label' or not e_inputs[k]: 
            continue
        if e_inputs[k][-1] not in string.punctuation:
            e_inputs[k] += '.'

    if partial_input:
        return ANLI_PARTIAL_INPUT.format(label=e_inputs['label'], hyp1=e_inputs['hyp1'], hyp2=e_inputs['hyp2'])
    
    return ANLI.format(**e_inputs)


def bucket_predictions(
    train_examples: List[AbductiveNLIExample],
    evaluation_examples: Dict[str, List[ParaphrasedAbductiveNLIExample]],
    train_fname: str,
    partial_input: bool = False,
):
    fasttext_classifier = FastTextClassifier(example_format_function=format_example, partial_input=partial_input)
    fasttext_classifier.train(train_examples, train_fname)

    buckets = {}
    for ex_id, paraphrases in tqdm(evaluation_examples.items()):        
        original_prediction, original_confidence = fasttext_classifier.infer(paraphrases[0].original_example)        
        
        bucket_confidences = []
        
        for p in paraphrases:
            para_prediction, para_confidence = fasttext_classifier.infer(p)
            
            bucket_confidences.append({
                'confidence': para_confidence.tolist(),
                'prediction': para_prediction,
                'paraphrased_example': asdict(p)
            })
        buckets[ex_id] = {
            'original_confidence': original_confidence.tolist(),
            'original_prediction': original_prediction,
            'gold_label': paraphrases[0].original_example.modeling_label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets, fasttext_classifier

def test_set_evaluation(examples, fasttext_model):
    predictions = []

    for example in tqdm(examples):
        prediction, confidence = fasttext_model.infer(example)

        predictions.append({
            'confidence': confidence.tolist(),
            'prediction': prediction,
            'label': example.modeling_label,
            'example_id': example.example_id
        })

    return predictions


if __name__ == '__main__':
    random.seed(42)
    
    training_examples = random.sample(anli_dataset.train_examples, len(anli_dataset.train_examples))

    ### Full input
    buckets, fasttext_model_full_input = bucket_predictions(
        training_examples, 
        anli_human, 
        os.path.join(PROJECT_ROOT_DIR, f'modeling/fasttext/abductive/training_data/fasttext_train_full_input.txt'),
        partial_input=False
    )
    write_json(buckets, f'modeling/fasttext/abductive/results/anli_human_full_input_lexical.json')

    test_set_preds_full_input = test_set_evaluation(anli_dataset.test_examples, fasttext_model_full_input)
    write_json(test_set_preds_full_input, f'modeling/fasttext/abductive/results/anli_test_set_full_input_lexical.json')

    ### Partial Input
    buckets, fasttext_model_partial_input = bucket_predictions(
        training_examples, 
        anli_human, 
        os.path.join(PROJECT_ROOT_DIR, f'modeling/fasttext/abductive/training_data/fasttext_train_partial_input.txt'),
        partial_input=True
    )
    write_json(buckets, f'modeling/fasttext/abductive/results/anli_human_partial_input_lexical.json')

    test_set_preds_partial_input = test_set_evaluation(anli_dataset.test_examples, fasttext_model_full_input)
    write_json(test_set_preds_partial_input, f'modeling/fasttext/abductive/results/anli_test_set__partial_input_lexical.json')
