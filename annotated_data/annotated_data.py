import os
import json
from typing import Dict, List, Tuple

from defeasible_data import (
    DefeasibleNLIDataset, 
    DefeasibleNLIExample, 
    ParaphrasedDefeasibleNLIExample
)

from abductive_data import (
    AbductiveNLIDataset,
    AbductiveNLIExample,
    ParaphrasedAbductiveNLIExample
)

from utils import load_json, PROJECT_ROOT_DIR, get_example_kv_pair_from_dict

def load_anli_paraphrased_set(validated_annotated_file: str) -> Dict[str, ParaphrasedAbductiveNLIExample]:
    """
    Loads approved paraphrases from a jsonl file.
    """
    validated_paraphrases = load_json(validated_annotated_file)
    return {
        e_id: [
            ParaphrasedAbductiveNLIExample(
                paraphrase_id=p['paraphrase_id'],
                original_example=AbductiveNLIExample(**p['original_example']),
                original_example_id=p['original_example_id'],
                hyp1_paraphrase=p['hyp1_paraphrase'],
                hyp2_paraphrase=p['hyp2_paraphrase'],
                worker_id=p['worker_id'],
                obs1_paraphrase=p['obs1_paraphrase'],
                obs2_paraphrase=p['obs2_paraphrase'],
                automatic_system_metadata=p['automatic_system_metadata'],
            ) for p in paraphrases
        ] for e_id, paraphrases in validated_paraphrases.items()
    }
    return validated_paraphrasess

def load_defeasible_paraphrased_set(validated_annotated_file: str) -> Dict[str, ParaphrasedDefeasibleNLIExample]:
    """
    Loads approved paraphrases from a jsonl file.
    """
    validated_paraphrases = load_json(validated_annotated_file)
    return {
        e_id: [
            ParaphrasedDefeasibleNLIExample(
                paraphrase_id=p['paraphrase_id'],
                original_example=DefeasibleNLIExample(**p['original_example']),
                original_example_id=p['original_example_id'],
                update_paraphrase=p['update_paraphrase'],
                worker_id=p['worker_id'],
                premise_paraphrase=p['premise_paraphrase'],
                hypothesis_paraphrase=p['hypothesis_paraphrase'],
                automatic_system_metadata=p['automatic_system_metadata'],
            ) for p in paraphrases
        ] for e_id, paraphrases in validated_paraphrases.items()
    }
    

dnli_social_human = load_defeasible_paraphrased_set('annotated_data/defeasible/social/social_paraphrases_human.json')
dnli_snli_human = load_defeasible_paraphrased_set('annotated_data/defeasible/snli/snli_paraphrases_human.json')
dnli_atomic_human = load_defeasible_paraphrased_set('annotated_data/defeasible/atomic/atomic_paraphrases_human.json')

dnli_human_dataset_by_name = {
    'snli': dnli_snli_human,
    'atomic': dnli_atomic_human,
    'social': dnli_social_human
}

anli_human = load_anli_paraphrased_set('annotated_data/abductive/anli_paraphrases_human.json')

para_nlu = {
    'anli': anli_human,
    'snli': dnli_snli_human,
    'atomic': dnli_atomic_human,
    'social': dnli_social_human,
}

para_nlu_pretty_names = {'anli': 'α-NLI', 'social': 'δ-SOCIAL', 'snli': 'δ-SNLI', 'atomic': 'δ-ATOMIC'}


