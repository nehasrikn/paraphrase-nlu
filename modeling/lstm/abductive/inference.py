from annotated_data.annotated_data import anli_human
from typing import List, Dict, Any, Tuple
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from abductive_data import anli_dataset
from tqdm import tqdm
import os
from dataclasses import asdict
import numpy as np
from utils import write_json, PROJECT_ROOT_DIR
from modeling.lstm.trained_model import TrainedLSTMModel
from modeling.lstm.generate_paranlu_data import form_abductive_example


def bucket_predictions(
    examples: Dict[str, List[ParaphrasedAbductiveNLIExample]], 
    nli_model: TrainedLSTMModel
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates predictions using nli_model for a family of paraphrases, as well as the original
    model.
    Each bucket:
        <original_example_id>: {
            'original_confidence': [confidence for original example],
            'original_prediction': [prediction for original example],
            'gold_label': [gold label for original example],
            'bucket_confidences': list of dicts (one for each paraphrase) (confidence, prediction, and paraphrased example)
        }

    """
    buckets = {}
    for ex_id, paraphrases in tqdm(examples.items()):
        
        original_prediction, original_confidence = nli_model.predict(
            *form_abductive_example(paraphrases[0].original_example, return_label=False)
        )
        bucket_confidences = []
        
        for p in paraphrases:
            prediction, confidence = nli_model.predict(*form_abductive_example(p, return_label=False))
            bucket_confidences.append({
                'confidence': confidence.tolist(),
                'prediction': prediction,
                'paraphrased_example': asdict(p)
            })
        
        buckets[ex_id] = {
            'original_confidence': original_confidence.tolist(),
            'original_prediction': original_prediction,
            'gold_label': paraphrases[0].original_example.modeling_label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets

def test_set_evaluation(examples, nli_model: TrainedLSTMModel):
    predictions = []

    for example in tqdm(examples):
        prediction, confidence = nli_model.predict(*form_abductive_example(example, return_label=False))
        predictions.append({
            'confidence': confidence.tolist(),
            'prediction': prediction,
            'label': example.modeling_label,
            'example_id': example.example_id
        })

    return predictions


if __name__ == '__main__':

    model = TrainedLSTMModel(os.path.join(PROJECT_ROOT_DIR, f'modeling/lstm/dataset/anli'))

    buckets = bucket_predictions(anli_human, model)
    test_set_predictions = test_set_evaluation(anli_dataset.test_examples, model)

    write_json(buckets, f'modeling/lstm/abductive/results/anli_human_bilstm.json')
    write_json(test_set_predictions, f'modeling/lstm/abductive/results/anli_test_set_bilstm.json')

    