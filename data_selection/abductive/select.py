import random
import os
from typing import List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import roundrobin
import json
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset, anli_dataset
from modeling.roberta.models import AbductiveTrainedModel
from collections import defaultdict
from dataclasses import asdict
import pandas as pd
from utils import plot_and_save_countplot, plot_and_save_hist, PROJECT_ROOT_DIR, termplot, write_json
from data_selection.data_selection_utils import stratify_examples_by_range, float_floor

def select_subset_random(data: AbductiveNLIDataset, num_examples: int = 115):
    """
    Unfortunately, we first selected 115 examples at random for the pilot :(
    This function replicates that. We then fill in the remaining examples to upsample
    examples predicted with weaker confidence.

    There's a heavy skew towards examples with high confidence, so stratified 
    sampling will need to downsample.
    """
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/roberta/abductive/chkpts/roberta-large-anli'), 
        multiple_choice=True
    )
    random.seed(42)
    sample = random.sample(data.get_split('test'), num_examples)
    confidences = [
        (e, roberta.predict(obs1=e.obs1, obs2=e.obs2, hyp1=e.hyp1, hyp2=e.hyp2)[e.modeling_label]) for e in tqdm(sample)
    ]

    stratified = {k: [e[0] for e in v] for k, v in stratify_examples_by_range(confidences, print_ranges=False).items()}
    return sample, stratified

def select_subset_by_stratified_confidence_hard_aflite_examples(
    data: AbductiveNLIDataset, 
    random_subset_already_selected: List[AbductiveNLIExample],
    confidence_ranges_of_random_subset: Dict[float, List[AbductiveNLIExample]],
    num_total_examples: int = 125
) -> List[AbductiveNLIExample]:
    """
    Selects a subset of examples from the test split of a dataset for analysis 
    (original examples to be paraphrased), or "buckets".
    We select examples to analyze that span the range of confidences a RoBERTa model has in the 
    example. 
    
    data: Original abductive dataset
    """
    example_ids_already_annotated = set([e.example_id for e in random_subset_already_selected])
    assert len(example_ids_already_annotated) == 115

    defacto_range_examples = 12

    stratified_examples = defaultdict(list)
    # for any given confidence range, we want to select ~12 examples, so if there's more, let's downsample
    for crange, crange_examples in confidence_ranges_of_random_subset.items():
        if len(crange_examples) > defacto_range_examples:
            stratified_examples[crange] = random.sample(crange_examples, defacto_range_examples)
        else:
            stratified_examples[crange] = crange_examples


    random.seed(42)
    test_split = data.get_split('test')
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/roberta/abductive/chkpts/roberta-large-anli'), 
        multiple_choice=True
    )

    # Get confidences for every single example in the test split
    test_split_confidences = [
        (e, roberta.predict(obs1=e.obs1, obs2=e.obs2, hyp1=e.hyp1, hyp2=e.hyp2)[e.modeling_label]) for e in tqdm(test_split) if e.example_id not in example_ids_already_annotated
    ]
    test_split_confidence_ranges = stratify_examples_by_range(test_split_confidences, print_ranges=False)
    for r in test_split_confidence_ranges.keys():
        random.seed(42)
        random.shuffle(test_split_confidence_ranges[r])

    assert len(test_split_confidence_ranges.keys()) == 10 # 10 confidence ranges
 
    # In this hard aflite split, we want 125 examples, i.e ~12 examples per confidence range    
    # Round robin style, iterate through all the confidence ranges, sampling one by one
    range_iterators = {range_floor: iter(range_examples) for range_floor, range_examples in test_split_confidence_ranges.items()}

    get_roundrobin_range_floor = roundrobin.basic(list(test_split_confidence_ranges.keys()))
    while sum([len(v) for v in stratified_examples.values()]) < num_total_examples:
        range_floor = get_roundrobin_range_floor()
        try:
            e = next(range_iterators[range_floor])
            stratified_examples[range_floor].append(e[0])
        except StopIteration as error:
            continue

    examples = []
    for k, v in stratified_examples.items():
        print(f'Confidence range {k}: {len(v)} examples')
        examples.extend(v)
    assert len(examples) == num_total_examples
    return examples

def select_subset_by_stratified_confidence_easy_aflite_examples(num_total_examples: int = 125):
    """
    Using examples that Chandra provided that are filtered out, select stratified by original model confidence
    InputSentence1 --> Same as obs1 in the public file. Contains Observation 1.
    InputSentence5 --> Same as obs2 in the public file. Contains Observation 2.
    RandomMiddleSentenceQuiz1: One candidate hypothesis.
    RandomMiddleSentenceQuiz2: Another candidate hypothesis.
    AnswerRightEnding: The label of which option is the correct hypothesis.
    """
    fname = 'raw-data/anli/af_filtered_out/train_easy_annotations.jsonl'
    easy_examples = []
    for i, json_str in enumerate(tqdm(list(open(fname, 'r')))):
        result = json.loads(json_str)
        easy_examples.append(AbductiveNLIExample(
            example_id='anli.train.easy.%d' % i,
            source_example_metadata=None,
            obs1=result['InputSentence1'],
            obs2=result['InputSentence5'],
            hyp1=result['RandomMiddleSentenceQuiz1'],
            hyp2 = result['RandomMiddleSentenceQuiz2'],
            label = int(result['AnswerRightEnding']),
            modeling_label=int(result['AnswerRightEnding'])-1,
            annotated_paraphrases=None
        ))
    
    confidences = []
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/roberta/abductive/chkpts/roberta-large-anli'), 
        multiple_choice=True
    )

    print('Loaded %d easy abductive examples' % len(easy_examples))

    random.seed(42)
    sample = random.sample(easy_examples, 5000)
    confidences = [
        (e, roberta.predict(obs1=e.obs1, obs2=e.obs2, hyp1=e.hyp1, hyp2=e.hyp2)[e.modeling_label]) for e in tqdm(sample)
    ]
    ranges = stratify_examples_by_range(confidences, print_ranges=False)
    assert len(ranges) == 10

    selected_examples = defaultdict(list)
    range_iterators = {range_floor: iter(range_examples) for range_floor, range_examples in ranges.items()}

    get_roundrobin_range_floor = roundrobin.basic(list(ranges.keys()))
    while sum([len(v) for v in selected_examples.values()]) < num_total_examples:
        range_floor = get_roundrobin_range_floor()
        try:
            e = next(range_iterators[range_floor])
            selected_examples[range_floor].append(e[0])
        except StopIteration as error:
            continue
    
    examples = []
    for k, v in selected_examples.items():
        print(f'Confidence range {k}: {len(v)} examples')
        print(v[0])
        examples.extend(v)
    assert len(examples) == num_total_examples
    return examples


if __name__ == '__main__':

    print('Recreating the random sample that was used for the pilot...')
    random_sample, confidence_ranges = select_subset_random(anli_dataset)

    write_json([asdict(e) for e in random_sample], 'data_selection/abductive/pilot_selected_examples.json')


    print('Upsampling and downsampling with the counts of random sample to get a stratified sample of 125 hard AFLITE examples...')
    random_and_stratified_examples = select_subset_by_stratified_confidence_hard_aflite_examples(
        anli_dataset, 
        random_sample, 
        confidence_ranges
    )

    print('Stratified sampling for 125 easy AFLITE examples...')
    easy_stratified_examples = select_subset_by_stratified_confidence_easy_aflite_examples()


    stratified_examples = random_and_stratified_examples + easy_stratified_examples
    # print(type(stratified_examples), type(stratified_examples[0]), stratified_examples[0])

    assert len(stratified_examples) == 250

    write_json([asdict(e) for e in stratified_examples], 'data_selection/abductive/selected_examples.json')