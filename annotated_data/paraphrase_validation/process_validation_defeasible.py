import json
import os
import numpy as np
from typing import List, Any, Dict
from dataclasses import asdict
from utils import load_jsonlines, PROJECT_ROOT_DIR, write_jsonlines, clean_paraphrase, write_json
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
import pandas as pd
from collections import defaultdict


def find_paraphrased_example_from_bucket_dict(paraphrase_id: str, bucket_dict: Dict[str, List]):
    """
    paraphrase_id: id of the paraphrase
    util function to find the paraphrased example from the bucket dict
    """
    original_example = ".".join(paraphrase_id.split('.')[:3]) # reconstruct original_example_id (#snli.train.18320.354P56DE9K32S5PUJB5MWGSOH19S7A.0)
    for p in bucket_dict[original_example] :
        if p['paraphrase_id'] == paraphrase_id:
            return p
    return None

def process_label_studio_validations(label_studio_validated_paraphrases: str) -> Dict[str, Any]:
    """
    label_studio_validated_paraphrases: path to jsonl file containing validated hits
    If paraphrase marked as invalid, it's invalid
    Paraphrases can be marked as valid but contain a minor edit to 
    ensure preservation of the crux of the reasoning problem 
    """
    valid = {}
    validation_annotation = pd.read_csv(label_studio_validated_paraphrases)
    print(validation_annotation.paraphrase_valid.value_counts(dropna=False, normalize=True))
    
    for _, a in validation_annotation.iterrows():
        if a.paraphrase_valid == 'invalid':
            continue
        valid[a.paraphrase_id] = a.paraphrase_edit if not pd.isnull(a.paraphrase_edit) else None

    print(len(valid)/len(validation_annotation))
    print(len([s for s in valid.values() if s is not None])/len(valid))
    return valid

def construct_gold_set_from_validation_annotation(
    all_paraphrases: str, 
    label_studio_validated_paraphrases: str,
    save_path: str
)-> Dict[str, List[ParaphrasedDefeasibleNLIExample]]:
    """
    Takes validated paraphrases through label studio and constructs a gold set
    """
    all_paraphrases = {e['example_id']: e['paraphrased_examples'] for e in load_jsonlines(all_paraphrases)}
    
    valid_paraphrases = process_label_studio_validations(label_studio_validated_paraphrases)
    
    gold_set = defaultdict(list)

    for paraphrase_id, paraphrase_edit in valid_paraphrases.items():
        paraphrase_example = find_paraphrased_example_from_bucket_dict(paraphrase_id, all_paraphrases)
        
        gold_set[paraphrase_example['original_example_id']].append(asdict(ParaphrasedDefeasibleNLIExample(
            paraphrase_id=paraphrase_id,
            original_example_id=paraphrase_example['original_example_id'],
            original_example=paraphrase_example['original_example'],
            update_paraphrase=paraphrase_edit if paraphrase_edit is not None else paraphrase_example['update_paraphrase'],
            worker_id=paraphrase_example['worker_id'],
            premise_paraphrase=paraphrase_example['premise_paraphrase'],
            hypothesis_paraphrase=paraphrase_example['hypothesis_paraphrase'],
            automatic_system_metadata=paraphrase_example['automatic_system_metadata']
        )))
    
    write_json(gold_set, save_path)
    return gold_set
        
        
if __name__ == '__main__':

    for dataset in ['snli', 'social', 'atomic']:
        construct_gold_set_from_validation_annotation(
            os.path.join(PROJECT_ROOT_DIR, f'mturk/defeasible/mturk_data/approved/{dataset}_approved.jsonl'),
            os.path.join(PROJECT_ROOT_DIR, f'annotated_data/paraphrase_validation/validation_annotation_files/human/dnli_{dataset}_mturk_validated.csv'),
            os.path.join(PROJECT_ROOT_DIR, f'annotated_data/defeasible/{dataset}/{dataset}_paraphrases_human.json')
        )



    #process_label_studio_validations(os.path.join(PROJECT_ROOT_DIR, 'annotated_data/paraphrase_validation/validation_annotation_files/human/dnli_snli_mturk_validated.csv'))
    #process_label_studio_validations(os.path.join(PROJECT_ROOT_DIR, 'annotated_data/paraphrase_validation/validation_annotation_files/human/dnli_social_mturk_validated.csv'))