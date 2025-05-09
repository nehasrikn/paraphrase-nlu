from modeling.fasttext.lexical_classifier import FastTextClassifier
from defeasible_data import DefeasibleNLIExample, DefeasibleNLIDataset, ParaphrasedDefeasibleNLIExample, dnli_datasets
from typing import Union, List, Any, Dict
from dataclasses import asdict
from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import PROJECT_ROOT_DIR, write_json, load_json
import os
from tqdm import tqdm
import string
import random


DNLI = "__label__{label} {premise} {hypothesis} {update}"
DNLI_PARTIAL_INPUT = "__label__{label} {update}"

def format_example(e: Union[DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample], partial_input=False):
    # todo preprocessing (remove stopwords etc)
    # make sure string endswith punctuation, or else add it
    e_inputs = {
        'premise': e.premise if isinstance(e, DefeasibleNLIExample) else e.original_example.premise,
        'hypothesis': e.hypothesis if isinstance(e, DefeasibleNLIExample) else e.original_example.hypothesis,
        'update': e.update if isinstance(e, DefeasibleNLIExample) else e.update_paraphrase,
        'label': e.label if isinstance(e, DefeasibleNLIExample) else e.original_example.label
    }
    for k in e_inputs:
        if k == 'label' or not e_inputs[k]: 
            continue
        if e_inputs[k][-1] not in string.punctuation:
            e_inputs[k] += '.'

    if partial_input:
        return DNLI_PARTIAL_INPUT.format(label=e_inputs['label'], update=e_inputs['update'])
    
    return DNLI.format(**e_inputs)


def bucket_predictions(
    train_examples: List[DefeasibleNLIExample],
    evaluation_examples: Dict[str, List[ParaphrasedDefeasibleNLIExample]],
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
            'gold_label': paraphrases[0].original_example.label,
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
            'label': example.label,
            'example_id': example.example_id
        })

    return predictions


if __name__ == '__main__':
    for dataset, evaluation_dataset in dnli_human_dataset_by_name.items():
        analysis_models_training_examples = [
            DefeasibleNLIExample(**e)
            for e in load_json(f'data_selection/defeasible/{dataset}/analysis_model_examples/train_examples.json')
        ]
        random.seed(42)
        analysis_models_training_examples = random.sample(analysis_models_training_examples, len(analysis_models_training_examples))

        ### Full input
        buckets, fasttext_model_full_input = bucket_predictions(
            analysis_models_training_examples, 
            evaluation_dataset, 
            os.path.join(PROJECT_ROOT_DIR, f'modeling/fasttext/defeasible/training_data/fasttext_train_full_input_{dataset}.txt'),
            partial_input=False
        )
        write_json(buckets, f'modeling/fasttext/defeasible/results/{dataset}/{dataset}_human_d-{dataset}-full_input_lexical.json')

        test_set_preds_full_input = test_set_evaluation(dnli_datasets[dataset].test_examples, fasttext_model_full_input)
        write_json(test_set_preds_full_input, f'modeling/fasttext/defeasible/results/{dataset}/{dataset}_test_set_d-{dataset}-full_input_lexical.json')

        ### Partial Input
        buckets, fasttext_model_partial_input = bucket_predictions(
            analysis_models_training_examples, 
            evaluation_dataset, 
            os.path.join(PROJECT_ROOT_DIR, f'modeling/fasttext/defeasible/training_data/fasttext_train_partial_input_{dataset}.txt'),
            partial_input=True
        )
        write_json(buckets, f'modeling/fasttext/defeasible/results/{dataset}/{dataset}_human_d-{dataset}-partial_input_lexical.json')

        test_set_preds_partial_input = test_set_evaluation(dnli_datasets[dataset].test_examples, fasttext_model_full_input)
        write_json(test_set_preds_partial_input, f'modeling/fasttext/defeasible/results/{dataset}/{dataset}_test_set_d-{dataset}-partial_input_lexical.json')
