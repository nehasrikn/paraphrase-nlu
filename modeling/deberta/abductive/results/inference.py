from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample, anli_dataset
from modeling.roberta.models import AbductiveTrainedModel
from annotated_data.annotated_data import anli_human
from tqdm import tqdm
import os
from dataclasses import asdict
import numpy as np
from utils import write_json, PROJECT_ROOT_DIR
from typing import List, Dict, Any, Tuple

from modeling.roberta.abductive.results.inference import bucket_predictions, test_set_evaluation

if __name__ == '__main__':
    deberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/deberta/abductive/chkpts/anli-deberta-v3-large'), 
        multiple_choice=True
    )

    write_json(
        bucket_predictions(anli_human, deberta),
        'modeling/deberta/abductive/results/anli_human_set_anli_deberta-v3-large.json'
    )
    write_json(test_set_evaluation(anli_dataset.test_examples, deberta), 'modeling/deberta/abductive/results/anli_test_set_anli_deberta-v3-large.json')
