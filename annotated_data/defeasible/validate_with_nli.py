from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Union, Tuple, List, Any
from scipy.special import softmax
import pandas as pd
import torch
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from string import punctuation
import json
from tqdm import tqdm

from utils import PROJECT_ROOT_DIR, PretrainedNLIModel

mnli_huggingface_labels = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

def plot_and_save(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.countplot(data=pd.DataFrame(values, columns=['value']), x="value")
    plt.xticks(rotation=45, ha='right')
    plot.get_figure().savefig(fig_file)





def nli_predict(premise: str, hypothesis: str, model: NLIModel, label_map=mnli_huggingface_labels) -> str:
    if not premise or not hypothesis:
        return None
    return label_map[np.argmax(model.predict(premise, hypothesis))]

def validate_mturk_hits_with_nli(approved_hits_path: str, nli_model: NLIModel):
    labels = []
    examples_with_labels = []
    with open(approved_hits_path) as f:
        approved_hits = json.load(f)
    print('Loaded %d approved hits' % len(approved_hits))
    for ex_id, paraphrased_examples in tqdm(approved_hits.items()):
        for i, p in enumerate(paraphrased_examples):
            forward = nli_predict(p['original_example']['update'], p['update_paraphrase'], nli_model)
            backward = nli_predict(p['update_paraphrase'], p['original_example']['update'], nli_model)
            labels.append(f'{forward[0]} {backward[0]}')
            examples_with_labels.append((p, (forward, backward)))
    
    return labels, examples_with_labels

if __name__ == '__main__':
    print('Loading model...')
    nli_model = NLIModel('roberta-large-mnli', cache_dir='/fs/clip-scratch/nehasrik/paraphrase-nlu/experiments/hf-cache/')
    labels, examples_with_labels = validate_mturk_hits_with_nli('atomic/atomic_approved_175.json', nli_model)
    plot_and_save(labels, 'atomic/nli_labels_validation.png')

    semantic_equivalence = defaultdict(list)
    for e, (forward, backward) in examples_with_labels:
        if forward == 'entailment' and backward == 'entailment':
            semantic_equivalence[e['original_example_id']].append(e)

    with open('atomic/semantic_equivalence.json', 'w') as f:
        json.dump(semantic_equivalence, f)