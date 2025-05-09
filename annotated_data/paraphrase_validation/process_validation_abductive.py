import json
import os
import numpy as np
from typing import List, Any, Dict
from dataclasses import asdict
from utils import load_jsonlines, PROJECT_ROOT_DIR, write_jsonlines, clean_paraphrase, write_json, load_json
from abductive_data import ParaphrasedAbductiveNLIExample
import pandas as pd
from collections import defaultdict

def find_paraphrased_example_from_bucket_dict(paraphrase_id: str, bucket_dict: Dict[str, List]):
    """
    paraphrase_id: id of the paraphrase
    util function to find the paraphrased example from the bucket dict
    """
    original_id_parts = 4 if 'easy' in paraphrase_id else 3  # (anli.train.easy.47662.3NGMS9VZTLI63TZEVA3ZD9MQ0DEFF0.1, anli.test.1471.3R5F3LQFV2K6EN37VPGW3VBM4DHZO0.2)
    original_example = ".".join(paraphrase_id.split('.')[:original_id_parts])
    for p in bucket_dict[original_example] :
        if p['paraphrase_id'] == paraphrase_id:
            return p
    return None

def process_label_studio_validations(label_studio_validated_paraphrases: str) -> Dict[str, Any]:
    """
    label_studio_validated_paraphrases: path to csv file containing validated hits
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
        valid[a.paraphrase_id] = {
            'hyp1': a.paraphrase_edit_hyp1 if not pd.isnull(a.paraphrase_edit_hyp1) else None,
            'hyp2': a.paraphrase_edit_hyp2 if not pd.isnull(a.paraphrase_edit_hyp2) else None,
        }
    
    print(len(valid)/len(validation_annotation))
    return valid

def construct_gold_set_from_validation_annotation(
    all_paraphrases: str, 
    label_studio_validated_paraphrases: str,
    save_path: str
)-> Dict[str, List[ParaphrasedAbductiveNLIExample]]:
    """
    Takes validated paraphrases through label studio and constructs a gold set
    We ended up annotating more examples than we needed, so trim the gold set to the number of examples we need
    to balance AF and non-AF examples.
    """
    all_paraphrases = {e['example_id']: e['paraphrased_examples'] for e in load_jsonlines(all_paraphrases)}
    selected_examples = [e['example_id'] for e in load_json('data_selection/abductive/selected_examples.json')]
    print(
        'Total original examples annotated (due to pilot first, then AF-based sampling later)', len(all_paraphrases),
        'Total examples in final abductive gold set', len(selected_examples)
    )
    
    valid_paraphrases = process_label_studio_validations(label_studio_validated_paraphrases)
    
    gold_set = defaultdict(list)

    for paraphrase_id, paraphrase_hyp_edits in valid_paraphrases.items():
        paraphrase_example = find_paraphrased_example_from_bucket_dict(paraphrase_id, all_paraphrases)
        assert paraphrase_example is not None
        
        gold_set[paraphrase_example['original_example_id']].append(asdict(ParaphrasedAbductiveNLIExample(
            paraphrase_id=paraphrase_id,
            original_example_id=paraphrase_example['original_example_id'],
            original_example=paraphrase_example['original_example'],
            hyp1_paraphrase=paraphrase_hyp_edits['hyp1'] if paraphrase_hyp_edits['hyp1'] is not None else paraphrase_example['hyp1_paraphrase'],
            hyp2_paraphrase=paraphrase_hyp_edits['hyp2'] if paraphrase_hyp_edits['hyp2'] is not None else paraphrase_example['hyp2_paraphrase'],
            worker_id=paraphrase_example['worker_id'],
            obs1_paraphrase=paraphrase_example['obs1_paraphrase'],
            obs2_paraphrase=paraphrase_example['obs2_paraphrase'],
            automatic_system_metadata=paraphrase_example['automatic_system_metadata']
        )))
    
    write_json(gold_set, f'annotated_data/abductive/anli_paraphrases_human_large.json')
    write_json(
        {k: v for k, v in gold_set.items() if k in selected_examples}, 
        f'annotated_data/abductive/anli_paraphrases_human.json'
    )
    return gold_set


if __name__ == '__main__':
    construct_gold_set_from_validation_annotation(
        os.path.join(PROJECT_ROOT_DIR, f'mturk/abductive/mturk_data/approved/anli_approved.jsonl'),
        os.path.join(PROJECT_ROOT_DIR, f'annotated_data/paraphrase_validation/validation_annotation_files/human/anli_validated.csv'),
        os.path.join(PROJECT_ROOT_DIR, f'annotated_data/abductive/anli_paraphrases_human.json')
    )