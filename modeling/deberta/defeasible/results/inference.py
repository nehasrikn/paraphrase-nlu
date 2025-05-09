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

if __name__ == '__main__':
    
    model_chkpt = 'modeling/deberta/defeasible/chkpts/d-{dataset_name}-deberta-v3-large'
    
    for dataset_name, dataset in dnli_human_dataset_by_name.items():
        dataset_specific_dnli_model = DefeasibleTrainedModel(
            os.path.join(PROJECT_ROOT_DIR, model_chkpt.format(dataset_name=dataset_name)), 
            '/fs/clip-projects/rlab/nehasrik/cache', 
            multiple_choice=False
        )
        
        buckets = bucket_predictions(dataset, dataset_specific_dnli_model)
        write_json(buckets, os.path.join(PROJECT_ROOT_DIR, f'modeling/deberta/defeasible/results/{dataset_name}/{dataset_name}_human_d-{dataset_name}-deberta-v3-large.json'))

        test_set_predictions_specialized = test_set_evaluation(dnli_datasets[dataset_name].test_examples, dataset_specific_dnli_model)
        write_json(test_set_predictions_specialized, f'modeling/deberta/defeasible/results/{dataset_name}/{dataset_name}_test_set_d-{dataset_name}-deberta-v3-large.json')