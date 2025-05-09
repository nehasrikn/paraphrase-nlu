from modeling.roberta.models import DefeasibleTrainedModel
from annotated_data.annotated_data import dnli_human_dataset_by_name
from typing import List, Dict, Any, Tuple
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample
from defeasible_data import dnli_datasets
from tqdm import tqdm
import os
from dataclasses import asdict
import numpy as np
from utils import write_json, PROJECT_ROOT_DIR

def bucket_predictions(
    examples: Dict[str, List[ParaphrasedDefeasibleNLIExample]], 
    nli_model: DefeasibleTrainedModel
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
            paraphrases[0].original_example.premise,
            paraphrases[0].original_example.hypothesis,
            paraphrases[0].original_example.update
        )
        original_binary_prediction = int(np.argmax(original_confidence))
        bucket_confidences = []
        
        for p in paraphrases:
            prediction = nli_model.predict(
                p.original_example.premise,
                p.original_example.hypothesis,
                p.update_paraphrase
            )
            bucket_confidences.append({
                'confidence': prediction.tolist(),
                'prediction': int(np.argmax(prediction)),
                'paraphrased_example': asdict(p)
            })
        
        buckets[ex_id] = {
            'original_confidence': original_confidence.tolist(),
            'original_prediction': original_binary_prediction,
            'gold_label': paraphrases[0].original_example.label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets

def test_set_evaluation(examples, nli_model: DefeasibleTrainedModel):
    predictions = []

    for example in tqdm(examples):
        confidence = nli_model.predict( #grabs original example object for first example in the list
            example.premise,
            example.hypothesis,
            example.update
        )
        predictions.append({
            'confidence': confidence.tolist(),
            'prediction': int(np.argmax(confidence)),
            'label': example.label,
            'example_id': example.example_id
        })

    return predictions
    

if __name__ == '__main__':
    
    model_chkpt = 'modeling/roberta/defeasible/chkpts/analysis_models/d-{dataset_name}-roberta-large'
    
    for dataset_name, dataset in dnli_human_dataset_by_name.items():
        dataset_specific_dnli_model = DefeasibleTrainedModel(
            os.path.join(PROJECT_ROOT_DIR, model_chkpt.format(dataset_name=dataset_name)), 
            '/fs/clip-projects/rlab/nehasrik/cache', 
            multiple_choice=False
        )
        
        buckets = bucket_predictions(dataset, dataset_specific_dnli_model)
        write_json(buckets, os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/results/{dataset_name}/{dataset_name}_human_d-{dataset_name}-roberta-large.json'))

        test_set_predictions_specialized = test_set_evaluation(dnli_datasets[dataset_name].test_examples, dataset_specific_dnli_model)
        write_json(test_set_predictions_specialized, f'modeling/roberta/defeasible/results/{dataset_name}/{dataset_name}_test_set_d-{dataset_name}-roberta-large.json')

        # general_dnli_model = DefeasibleTrainedModel(
        #     os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/chkpts/roberta-large-dnli'), 
        #     'experiments/hf-cache', 
        #     multiple_choice=False
        # )

        general_dnli_buckets = bucket_predictions(dataset, general_dnli_model)
        write_json(general_dnli_buckets, os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/results/{dataset_name}/{dataset_name}_human_dnli-roberta-large.json'))

        #test_set_predictions_general = test_set_evaluation(dnli_datasets[dataset_name].test_examples, general_dnli_model)
        #write_json(test_set_predictions_general, f'modeling/roberta/defeasible/results/{dataset_name}/{dataset_name}_test_set_dnli-roberta-large.json')
