from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample, anli_dataset
from modeling.roberta.models import AbductiveTrainedModel
from annotated_data.annotated_data import anli_human
from tqdm import tqdm
import os
from dataclasses import asdict
import numpy as np
from utils import write_json, PROJECT_ROOT_DIR
from typing import List, Dict, Any, Tuple

def bucket_predictions(
    examples: Dict[str, List[ParaphrasedAbductiveNLIExample]], 
    nli_model: AbductiveTrainedModel
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
        
        original_confidence = nli_model.predict( #grabs original example object for first example in the list
            obs1=paraphrases[0].original_example.obs1,
            obs2=paraphrases[0].original_example.obs2,
            hyp1=paraphrases[0].original_example.hyp1,
            hyp2=paraphrases[0].original_example.hyp2
        )

        original_binary_prediction = int(np.argmax(original_confidence))
        bucket_confidences = []
        
        for p in paraphrases:
            prediction = nli_model.predict(
                obs1=p.original_example.obs1,
                obs2=p.original_example.obs2,
                hyp1=p.hyp1_paraphrase,
                hyp2=p.hyp2_paraphrase
            )
            bucket_confidences.append({
                'confidence': prediction.tolist(),
                'prediction': int(np.argmax(prediction)),
                'paraphrased_example': asdict(p)
            })
        
        buckets[ex_id] = {
            'original_confidence': original_confidence.tolist(),
            'original_prediction': original_binary_prediction,
            'gold_label': paraphrases[0].original_example.modeling_label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets


def test_set_evaluation(examples, nli_model: AbductiveTrainedModel):
    predictions = []

    for example in tqdm(examples):
        confidence = nli_model.predict(
            obs1=example.obs1, obs2=example.obs2, hyp1=example.hyp1, hyp2=example.hyp2
        )
        predictions.append({
            'confidence': confidence.tolist(),
            'prediction': int(np.argmax(confidence)),
            'label': example.modeling_label,
            'example_id': example.example_id
        })
    return predictions



if __name__ == '__main__':
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/roberta/abductive/chkpts/roberta-large-anli'), 
        multiple_choice=True
    )

    # write_json(test_set_evaluation(anli_dataset.test_examples, roberta), 'modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json')
    write_json(
        bucket_predictions(anli_human, roberta),
        'modeling/roberta/abductive/results/anli_human_set_anli_roberta-large.json'
    )