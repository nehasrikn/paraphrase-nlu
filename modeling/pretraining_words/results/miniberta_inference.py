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

from modeling.roberta.defeasible.results.inference import bucket_predictions, test_set_evaluation


config = {
    '10M-2': {
        'name': 'roberta-base-10M-2', 
        'location': 'modeling/pretraining_words/minibertas/10M-pretraining-words/d-{dataset}-roberta-base-10M-2',
        'results': 'modeling/pretraining_words/results/{dataset}/{dataset}_human_d-{dataset}-roberta-base-10M-2.json',
        'test_results': 'modeling/pretraining_words/results/{dataset}/{dataset}_test_set_d-{dataset}-roberta-base-10M-2.json'
    },
    '100M-2': {
        'name': 'roberta-base-100M-2', 
        'location': 'modeling/pretraining_words/minibertas/100M-pretraining-words/d-{dataset}-roberta-base-100M-2',
        'results': 'modeling/pretraining_words/results/{dataset}/{dataset}_human_d-{dataset}-roberta-base-100M-2.json',
        'test_results': 'modeling/pretraining_words/results/{dataset}/{dataset}_test_set_d-{dataset}-roberta-base-100M-2.json'
    },
    '1B-3': {
        'name': 'roberta-base-1B-3', 
        'location': 'modeling/pretraining_words/minibertas/1B-pretraining-words/d-{dataset}-roberta-base-1B-3',
        'results': 'modeling/pretraining_words/results/{dataset}/{dataset}_human_d-{dataset}-roberta-base-1B-3.json',
        'test_results': 'modeling/pretraining_words/results/{dataset}/{dataset}_test_set_d-{dataset}-roberta-base-1B-3.json'
        
    },
    'base': {
        'name': 'roberta-base', 
        'location': 'modeling/roberta/defeasible/chkpts/analysis_models/d-{dataset}-roberta-base',
        'results': 'modeling/roberta/defeasible/results/{dataset}/{dataset}_human_d-{dataset}-roberta-base.json',
        'test_results': 'modeling/roberta/defeasible/results/{dataset}/{dataset}_test_set_d-{dataset}-roberta-base.json'
    },
    
    '1M-2': {
        'name': 'roberta-med-small-1M-2', 
        'location': 'modeling/pretraining_words/minibertas/1M-pretraining-words/d-{dataset}-roberta-med-small-1M-2',
        'results': 'modeling/pretraining_words/results/{dataset}/{dataset}_human_d-{dataset}-roberta-med-small-1M-2.json',
        'test_results': 'modeling/pretraining_words/results/{dataset}/{dataset}_test_set_d-{dataset}-roberta-med-small-1M-2.json'
    },
}

if __name__ == '__main__':
    
    for dataset_name, dataset in dnli_human_dataset_by_name.items():
        print("Running miniberta inference for dataset:", dataset_name)
        for size, info in config.items():
            print("Size:", size)
            miniberta = DefeasibleTrainedModel(
                info['location'].format(dataset=dataset_name),
                '/fs/clip-projects/rlab/nehasrik/cache', 
                multiple_choice=False
            )
            
            buckets = bucket_predictions(dataset, miniberta)
            write_json(buckets, info['results'].format(dataset=dataset_name))

            test_set_predictions_specialized = test_set_evaluation(dnli_datasets[dataset_name].test_examples, miniberta)
            write_json(test_set_predictions_specialized, info['test_results'].format(dataset=dataset_name))

    
    
    
    
    
        