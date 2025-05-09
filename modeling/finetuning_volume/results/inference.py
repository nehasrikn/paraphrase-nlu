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
    
    for dataset_name, dataset in dnli_human_dataset_by_name.items():
        for size in ["0.01", "0.05", "0.1", "0.5"]:
            miniberta = DefeasibleTrainedModel(
                os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/chkpts/analysis_models/pretraining-vs-finetuning/varying-finetuning/{dataset_name}/d-{dataset_name}-roberta-base-{size}'),
                '/fs/clip-projects/rlab/nehasrik/cache', 
                multiple_choice=False
            )
            
            buckets = bucket_predictions(dataset, miniberta)
            write_json(buckets, f'modeling/roberta/defeasible/varying-finetuning/results/{dataset_name}/{dataset_name}_human_d-{dataset_name}-roberta-base-{size}.json')

            test_set_predictions_specialized = test_set_evaluation(dnli_datasets[dataset_name].test_examples, miniberta)
            write_json(test_set_predictions_specialized, f'modeling/roberta/defeasible/varying-finetuning/results/{dataset_name}/{dataset_name}_test_d-{dataset_name}-roberta-base-{size}.json')

    
    
    
    
        