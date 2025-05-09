import json
import os
import numpy as np
from typing import List, Any, Dict
from utils import load_jsonlines, PROJECT_ROOT_DIR, write_jsonlines, clean_paraphrase
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
from abductive_data import anli_dataset, AbductiveNLIExample, ParaphrasedAbductiveNLIExample
import pandas as pd
from collections import defaultdict


def dedup_bucket(paraphrases: str, task='defeasible') -> Dict[str, List[Dict]]:
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    util function to dedup paraphrases in a bucket before validation so as not to waste
    annotator time.
    """
    paraphrase_buckets = defaultdict(list)
    for bucket in load_jsonlines(paraphrases):
        bucket_paraphrases = set()
        for paraphrase in bucket['paraphrased_examples']:
            if task == 'defeasible':
                cleaned = clean_paraphrase(paraphrase['update_paraphrase'])
            else:
                cleaned = clean_paraphrase(paraphrase['hyp1_paraphrase']) + ' ' + clean_paraphrase(paraphrase['hyp2_paraphrase'])

            if cleaned in bucket_paraphrases:
                continue

            if task == 'defeasible':
                paraphrase_buckets[bucket['example_id']].append(ParaphrasedDefeasibleNLIExample(**paraphrase))
            else:
                paraphrase_buckets[bucket['example_id']].append(ParaphrasedAbductiveNLIExample(**paraphrase))

            bucket_paraphrases.add(cleaned)

    return paraphrase_buckets

def construct_defeasible_doc_string(paraphrase: ParaphrasedDefeasibleNLIExample) -> str:
    form_str = f"""Paraphrase ID: {paraphrase.paraphrase_id}

    Premise: {paraphrase.original_example['premise']}
    Hypothesis: {paraphrase.original_example['hypothesis']}
    Update: {paraphrase.original_example['update']}
    Update Type: {paraphrase.original_example['update_type']}

    Paraphrase: {paraphrase.update_paraphrase}
    """
    return form_str

def construct_abductive_doc_string(paraphrase: ParaphrasedAbductiveNLIExample) -> str:
    form_str = f"""Paraphrase ID: {paraphrase.paraphrase_id}

    Obs1: {paraphrase.original_example['obs1']}
    Obs2: {paraphrase.original_example['obs2']}
    Hypothesis 1: {paraphrase.original_example['hyp1']}
    Hypothesis 2: {paraphrase.original_example['hyp2']}
    Label: {paraphrase.original_example['label']}

    Hyp 1 Paraphrase: {paraphrase.hyp1_paraphrase}
    Hyp 2 Paraphrase: {paraphrase.hyp2_paraphrase}
    """
    return form_str


def export_defeasible_approved_paraphrases_to_label_studio_format(
    paraphrases: str,
    outfile: str
):
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    """
    deduped_buckets = dedup_bucket(paraphrases, task='defeasible')

    examples = []
    for original_example_id, bucket in deduped_buckets.items():
        for paraphrase in bucket:
            form_str = construct_defeasible_doc_string(paraphrase)
            examples.append({
                'paraphrase_example': form_str,
                'original_example_id': paraphrase.original_example['example_id'],
                'paraphrase_id': paraphrase.paraphrase_id,
            })
    
    pd.DataFrame(examples).to_csv(outfile, index=False)

def export_abductive_approved_paraphrases_to_label_studio_format(
    paraphrases: str,
    outfile: str
):
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    """
    deduped_buckets = dedup_bucket(paraphrases, task='abductive')

    examples = []
    for original_example_id, bucket in deduped_buckets.items():
        for paraphrase in bucket:
            form_str = construct_abductive_doc_string(paraphrase)
            examples.append({
                'paraphrase_example': form_str,
                'original_example_id': paraphrase.original_example['example_id'],
                'paraphrase_id': paraphrase.paraphrase_id,
            })
    
    pd.DataFrame(examples).to_csv(outfile, index=False)

def export_mturk_defeasible_data():
    export_defeasible_approved_paraphrases_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/snli_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/snli_approved.csv'
    )

    export_defeasible_approved_paraphrases_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/social_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/social_approved.csv'
    )

    export_defeasible_approved_paraphrases_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/atomic_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/atomic_approved.csv'
    )



export_abductive_approved_paraphrases_to_label_studio_format(
    'mturk/abductive/mturk_data/approved/anli_approved.jsonl',
    'annotated_data/paraphrase_validation/source_files/human/anli_approved.csv'
)