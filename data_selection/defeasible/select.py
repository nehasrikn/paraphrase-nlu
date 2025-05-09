import random
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import roundrobin
import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr
from defeasible_data import DefeasibleNLIExample, DefeasibleNLIDataset
from modeling.roberta.models import DefeasibleTrainedModel
from collections import defaultdict
from dataclasses import asdict
from data_selection.data_selection_utils import float_floor, stratify_examples_by_range

# python -m data_selection.defeasible.select

def select_train_dev_set_for_aflite_embedding_model(
    data: DefeasibleNLIDataset,
    out_dir: str,
    frac_train_examples: float = 0.10,
    min_train_examples: int = 5000,
    min_dev_examples: int = 1000
) -> Tuple[List[DefeasibleNLIExample],List[DefeasibleNLIExample]]:
    """
    Selects a subset of train and dev examples to train an embedding model
    for AFLite.
    """

    random.seed(42)

    def incrementally_sample_by_ph_id(split: str, num_examples: int) -> List[DefeasibleNLIExample]:
        random.seed(42)
        ph_ids = data.get_split_premise_hypothesis_ids(split) # returns ordered list
        #shuffle list and then peel off premi-hyp pairs as needed
        random.shuffle(ph_ids)
        examples = []
        ph_idx = 0
        while len(examples) < num_examples:
            examples.extend(data.get_examples_for_premise_hypothesis(ph_ids[ph_idx]))
            ph_idx += 1
        print('Used %d of %d P-H pairs...' % (ph_idx, len(ph_ids)))
        return examples

    num_train_examples = max(int(frac_train_examples * len(data.train_examples)), min_train_examples)
    print('Sampling %d train examples...' % num_train_examples)
    train_examples = incrementally_sample_by_ph_id('train', num_train_examples) 
    
    print('Sampling %d dev examples...' % min_dev_examples)
    dev_examples = incrementally_sample_by_ph_id('dev', min_dev_examples)


    DefeasibleNLIDataset.write_processed_examples_for_modeling(train_examples, out_dir=out_dir, fname='aflite_train.csv')
    DefeasibleNLIDataset.write_processed_examples_for_modeling(dev_examples, out_dir=out_dir, fname='aflite_dev.csv')

    with open(os.path.join(out_dir, 'aflite_train_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in train_examples]))
    
    with open(os.path.join(out_dir, 'aflite_dev_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in dev_examples]))

    return (train_examples, dev_examples)

def run_select_train_dev_set_for_aflite_embedding_model():
    for data_source in ['social', 'atomic', 'snli']:
        print('#### Generating AFLite embedding train/dev set for %s ####' % data_source)
        aflite_data_generation = {
            'data': DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/', data_source),
            'out_dir': f'data_selection/defeasible/{data_source}',
            'frac_train_examples': 0.10,
            'min_train_examples': 5000,
            'min_dev_examples': 1000,
        }
        select_train_dev_set_for_aflite_embedding_model(**aflite_data_generation)
        print()

def plot_and_save(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.histplot(data=pd.DataFrame(values, columns=['value']), x="value", kde=True)
    plot.get_figure().savefig(fig_file)

def select_stratify_examples_by_range(ranges: Dict[float, List[Tuple[DefeasibleNLIExample, float]]], num_total_examples: int=125) -> List[Any]:
    """
    Helper function to turn lists of example ids and scores into a dictionary [(ex_1, 0.43), (ex_2, 0.02), ...] -> {0: [ex_2], 0.4: [ex_1], ...}
    and then samples num_examples_per_range.
    If there are less than num_examples_per_range, then all examples for that range are selected.
    """
    range_iterators = {range_floor: iter(range_examples) for range_floor, range_examples in ranges.items()}
    examples = []
    example_confidences = []
    
    get_roundrobin_range_floor = roundrobin.basic(list(ranges.keys()))
    while len(examples) < num_total_examples:
        range_floor = get_roundrobin_range_floor()
        try:
            e = next(range_iterators[range_floor])
            examples.append(e[0]) # only appending the actual example, not the score!
            example_confidences.append(range_floor)
        except StopIteration as error:
            continue

    print(pd.Series(example_confidences).value_counts())
    return examples
    
def compare_aflite_variances_with_analysis_model_confidence(
    examples: List[DefeasibleNLIExample],
    analysis_model: DefeasibleTrainedModel,
    aflite_scores:  Dict[str, List[float]],
    fig_dir: str
):
    ### Inspect AFLite variances for selected examples (plot variance vs analysis model confidence)
    
    variances = [np.var(aflite_scores[e.example_id]) for e in examples]
    analysis_model_confidences = generate_confidence_distribution_of_analysis_model_on_sample(
        analysis_model, 
        examples,
        do_plot_and_save=True,
        fig_file=os.path.join(fig_dir, 'selected_examples_analysis_model_confidences.png'),
    )

    plt.figure()
    plt.title('AFLite Variance Across Iterations vs. RoBERTa Analysis Model Confidence')
    plt.xlabel('aflite_variance')
    plt.ylabel('roberta_analysis_model_confidence')
    plot = sns.scatterplot(x=variances, y=analysis_model_confidences)
    plot.get_figure().savefig(os.path.join(fig_dir, 'aflite_variances_vs_analysis_model_confidences.png'))

def select_data_for_annotation(
    example_pool: List[Tuple[DefeasibleNLIExample, float]], 
    analysis_model: DefeasibleTrainedModel,
    aflite_scores:  Dict[str, List[float]],
    fig_dir: str,
    easy_partition_threshold: float=0.75
)-> List[DefeasibleNLIExample]: 
    easy_partition, hard_partition = [], []

    for e, example_af_scores in example_pool:
        if np.mean(example_af_scores) > easy_partition_threshold:
            easy_partition.append(DefeasibleNLIExample(**e))
        else:
            hard_partition.append(DefeasibleNLIExample(**e))

    print('%d easy examples & %d hard examples' % (len(easy_partition), len(hard_partition)))
    
    easy_analysis_model_confidences = generate_confidence_distribution_of_analysis_model_on_sample(
        analysis_model, 
        easy_partition, 
        do_plot_and_save=True,
        fig_file=os.path.join(fig_dir, 'aflite_easy_example_sample_analysis_model_confidences.png'),
    )
    easy_analysis_model_confidences_stratified = stratify_examples_by_range(list(zip(easy_partition, easy_analysis_model_confidences)))
    selected_easy_examples = select_stratify_examples_by_range(easy_analysis_model_confidences_stratified)

    hard_analysis_model_confidences = generate_confidence_distribution_of_analysis_model_on_sample(
        analysis_model, 
        hard_partition, 
        do_plot_and_save=True,
        fig_file=os.path.join(fig_dir, 'aflite_hard_example_sample_analysis_model_confidences.png'),
    )
    hard_analysis_model_confidences_stratified = stratify_examples_by_range(list(zip(hard_partition, hard_analysis_model_confidences)))
    selected_hard_examples = select_stratify_examples_by_range(hard_analysis_model_confidences_stratified)

    selected_examples = selected_easy_examples + selected_hard_examples

    compare_aflite_variances_with_analysis_model_confidence(selected_examples, analysis_model, aflite_scores, fig_dir)
    print('Done selecting %d examples!' % len(selected_examples))
    with open(os.path.join(fig_dir, 'selected_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in selected_examples]))

def select_data_for_train_analysis_model(
    af_scores: Dict[str, List[float]], 
    dataset: DefeasibleNLIDataset,
    fig_dir: str,
    out_dir: str, 
    agg_function=np.median,
    frac_annotation_sample=0.055
)-> Tuple[List[DefeasibleNLIExample], List[DefeasibleNLIExample]]:
    """
    Partitions train set into two splits: one pool from which we can select examples for annotation, and another pool
    that is used to train the analysis model. We don't select the subset of examples in this particular 
    """
    random.seed(42)
    score_aggregate_values = [] # aggregate scores per example across each outer iteration of adversarial filtering
    score_lengths = [] # get number of iterations this particular example was in before being filtered out
    ph_id_lookup = defaultdict(list) # so that we can split by ph-id (sample by ph-id, and get all associated)
    
    for k, v in af_scores.items(): #k = example_id, v = list of scores
        ph_id_lookup[dataset.get_example_by_id(k).premise_hypothesis_id].append((k, v))
        score_aggregate_values.append(agg_function(v))
        score_lengths.append(len(v))

    plot_and_save(score_aggregate_values, os.path.join(fig_dir, 'aflite_score_distribution_%s.png' % agg_function.__name__))
    plot_and_save(score_lengths, os.path.join(fig_dir, 'aflite_score_distribution_lengths.png'))
    
    shuffled_ph_ids = random.sample(list(ph_id_lookup.keys()), len(list(ph_id_lookup.keys()))) #shuffle examples in place

    num_ph_ids_for_annotation = int(frac_annotation_sample * len(shuffled_ph_ids))
    print(f'Sampling {num_ph_ids_for_annotation} premise-hypotheses for annotation...')

    annotation_example_ids = []
    for ph_id in shuffled_ph_ids[:num_ph_ids_for_annotation]: #peel off associated examples of in-place shuffled ph-ids
        annotation_example_ids.extend(ph_id_lookup[ph_id])

    plot_and_save([agg_function(v) for k, v in annotation_example_ids], os.path.join(fig_dir, 'aflite_score_distribution_annotation_sample_%s.png' % agg_function.__name__))

    analysis_example_ids = []
    for ph_id in shuffled_ph_ids[num_ph_ids_for_annotation:]: #get remaining non-annotation examples as analysis model training examples
        analysis_example_ids.extend(ph_id_lookup[ph_id])

    plot_and_save([agg_function(v) for k, v in analysis_example_ids], os.path.join(fig_dir, 'aflite_score_distribution_annotation_analysis_model_%s.png' % agg_function.__name__))

    print('Sampled %d examples for annotation stratified selection and %d for training analysis model' % (len(annotation_example_ids), len(analysis_example_ids)))
    with open(os.path.join(out_dir, 'annotation_examples/annotation_example_pool_ph_id.json'), "w") as file:
        file.write(json.dumps([(asdict(dataset.get_example_by_id(e)), score) for e, score in annotation_example_ids]))

    analysis_examples = [dataset.get_example_by_id(e) for e, score in analysis_example_ids]
    DefeasibleNLIDataset.write_processed_examples_for_modeling(analysis_examples, out_dir=out_dir, fname='analysis_model_examples/train_examples.csv')
    with open(os.path.join(out_dir, 'analysis_model_examples/train_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in analysis_examples]))

    print('%d annotation examples, %d analysis model training examples' % (len(annotation_example_ids), len(analysis_examples)))
    return analysis_examples

def generate_confidence_distribution_of_analysis_model_on_sample(
    model: DefeasibleTrainedModel, 
    annotation_set: List[DefeasibleNLIExample],
    do_plot_and_save: bool = True,
    fig_file: Optional[str] = None,
) -> None:

    confidences = []
    for e in tqdm(annotation_set):
        prediction = model.predict(premise=e.premise, hypothesis=e.hypothesis, update=e.update)
        confidences.append(prediction[e.label])
    if do_plot_and_save:
        plot_and_save(confidences, fig_file)
    return np.array(confidences)
    
def run_compare_analysis_model_confidence_and_aflite_aggregated_score():
    for data_source in ['atomic', 'social', 'snli']:
        model = DefeasibleTrainedModel(f'modeling/defeasible/chkpts/analysis_models/d-{data_source}-roberta-large', multiple_choice=False)
        annotation_set = [
            DefeasibleNLIExample(**j)
            for j in json.load(open(f'data_selection/defeasible/{data_source}/annotation_examples/selected_stratified_annotation_examples.json', 'rb'))
        ]
        analysis_model_confidences = generate_confidence_distribution_of_analysis_model_on_sample(model, annotation_set, f'data_selection/defeasible/{data_source}/annotation_examples/analysis_model_confidence.png')
        aflite_scores = json.load(open(f'data_selection/aflite/{data_source}/{data_source}_af_scores.json'))
        aflite_agg_scores = np.array([np.mean(aflite_scores[e.example_id]) for e in annotation_set])

        print(len(aflite_agg_scores), len(analysis_model_confidences))
        print('Pearson correlation: ', pearsonr(analysis_model_confidences, aflite_agg_scores))

        plt.figure()
        plot = sns.scatterplot(x=aflite_agg_scores, y=analysis_model_confidences)
        plot.get_figure().savefig(f'data_selection/defeasible/{data_source}/annotation_examples/analysis_model_confidence_vs_aflite_score.png')
        
def run_select_data_for_annotation():
    for data_source in ['atomic', 'social', 'snli']:
        print('################### %s ###################' % data_source)
        dnli_dataset = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/', data_source)
        out_dir = f'data_selection/defeasible/{data_source}'
        annotation_example_pool = json.load(open(
            f'data_selection/defeasible/{data_source}/annotation_examples/annotation_example_pool_ph_id.json'
        ))
        model = DefeasibleTrainedModel(f'modeling/defeasible/chkpts/analysis_models/d-{data_source}-roberta-large', multiple_choice=False)
        aflite_scores = json.load(open(f'data_selection/aflite/{data_source}/{data_source}_af_scores.json'))
        select_data_for_annotation(annotation_example_pool, model, aflite_scores=aflite_scores, fig_dir=f'data_selection/defeasible/{data_source}/annotation_examples/')

def run_select_data_for_train_analysis_model():
    for data_source in ['atomic', 'social', 'snli']:
        print('################### %s ###################' % data_source)
        dnli_dataset = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/', data_source)
        af_scores = json.load(open(f'data_selection/aflite/{data_source}/{data_source}_af_scores.json'))
        out_dir = f'data_selection/defeasible/{data_source}'

        select_data_for_train_analysis_model(
            af_scores, 
            dnli_dataset, 
            fig_dir=f'data_selection/aflite/{data_source}/figures', 
            out_dir=out_dir,
            agg_function=np.mean
        )
        # Dev examples will be those not used for AFLite Embedding model evaluation
        aflite_embedding_dev_examples = json.load(open(f'data_selection/defeasible/{data_source}/aflite_embedding_model_examples/aflite_dev_examples.json'))
        aflite_embedding_dev_example_ids = set(e['example_id'] for e in aflite_embedding_dev_examples)

        analysis_dev_examples = [e for e in dnli_dataset.get_split('dev') if e.example_id not in aflite_embedding_dev_example_ids]
        with open(os.path.join(out_dir, 'analysis_model_examples/dev_examples.json'), "w") as file:
            file.write(json.dumps([asdict(e) for e in analysis_dev_examples]))
       
        DefeasibleNLIDataset.write_processed_examples_for_modeling(
            analysis_dev_examples, 
            out_dir=out_dir, 
            fname='analysis_model_examples/dev_examples.csv'
        )

if __name__ == '__main__':
    random.seed(42)

    # run_select_train_dev_set_for_aflite_embedding_model()

    #run_select_data_for_train_analysis_model()

    #run_compare_analysis_model_confidence_and_aflite_aggregated_score()
    run_select_data_for_annotation()

    