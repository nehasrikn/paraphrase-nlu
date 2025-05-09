from abductive_data import anli_dataset, AbductiveNLIExample, ParaphrasedAbductiveNLIExample
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, dnli_datasets, ParaphrasedDefeasibleNLIExample
from nltk.tokenize import word_tokenize
from typing import List, Set, Union
from tqdm import tqdm
from utils import load_json, write_strings_to_file
import string

DEFEASIBLE_S1_EXAMPLE_TEMPLATE = "{premise} {hypothesis}"
DEFEASIBLE_S2_EXAMPLE_TEMPLATE = "{update}"

ABDUCTIVE_S1_EXAMPLE_TEMPLATE = "{obs1} {hyp1} {obs2}"
ABDUCTIVE_S2_EXAMPLE_TEMPLATE = "{obs1} {hyp2} {obs2}"

def tokenize(text: str) -> List[str]:
    if text and text[-1] not in string.punctuation:
        text += '.'
    return " ".join(word_tokenize(text))

def form_defeasible_example(ex: Union[DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample], return_label=True) -> str:
    if isinstance(ex, ParaphrasedDefeasibleNLIExample):
        premise = ex.original_example.premise
        hypothesis = ex.original_example.hypothesis
        update = ex.update_paraphrase
    elif isinstance(ex, DefeasibleNLIExample):
        premise = ex.premise
        hypothesis = ex.hypothesis
        update = ex.update
    else:
        raise ValueError("ex must be either a DefeasibleNLIExample or a ParaphrasedDefeasibleNLIExample")

    s1 = DEFEASIBLE_S1_EXAMPLE_TEMPLATE.format(
        premise=tokenize(premise), hypothesis=tokenize(hypothesis)
    )
    s2 = DEFEASIBLE_S2_EXAMPLE_TEMPLATE.format(update=tokenize(update))
    if return_label:
        return s1, s2, ex.update_type

    return s1, s2


def form_abductive_example(ex: Union[AbductiveNLIExample, ParaphrasedAbductiveNLIExample], return_label=True) -> str:
    if isinstance(ex, ParaphrasedAbductiveNLIExample):
        obs1 = ex.original_example.obs1
        obs2 = ex.original_example.obs2
        hyp1 = ex.hyp1_paraphrase
        hyp2 = ex.hyp2_paraphrase
    elif isinstance(ex, AbductiveNLIExample):
        obs1 = ex.obs1
        obs2 = ex.obs2
        hyp1 = ex.hyp1
        hyp2 = ex.hyp2

    s1 = ABDUCTIVE_S1_EXAMPLE_TEMPLATE.format(
        obs1=tokenize(obs1), hyp1=tokenize(hyp1), obs2=tokenize(obs2)
    )
    s2 = ABDUCTIVE_S2_EXAMPLE_TEMPLATE.format(
        obs1=tokenize(obs1), hyp2=tokenize(hyp2), obs2=tokenize(obs2)
    )
    if return_label:
        return s1, s2, ex.modeling_label

    return s1, s2

def examples_to_data_lists(examples, dataset_name: str, split: str, form_func=form_defeasible_example):
    s1_data, s2_data, labels = [], [], []
    for example in tqdm(examples):
        s1, s2, label = form_func(example)
        s1_data.append(s1)
        s2_data.append(s2)
        labels.append(label)

    write_strings_to_file(s1_data, f"modeling/lstm/dataset/{dataset_name}/s1.{split}")
    write_strings_to_file(s2_data, f"modeling/lstm/dataset/{dataset_name}/s2.{split}")
    write_strings_to_file(labels, f"modeling/lstm/dataset/{dataset_name}/labels.{split}")

def write_training_data_defeasible():
    for dataset_name, dataset in dnli_datasets.items():
        examples_to_data_lists([
            DefeasibleNLIExample(**e)
            for e in load_json(f'data_selection/defeasible/{dataset_name}/analysis_model_examples/train_examples.json')
        ], f'd-{dataset_name}', 'train')
        examples_to_data_lists(dataset.dev_examples, f'd-{dataset_name}', 'dev')
        examples_to_data_lists(dataset.test_examples, f'd-{dataset_name}', 'test')

def write_training_data_abductive():
    examples_to_data_lists(anli_dataset.train_examples, f'anli', 'train', form_func=form_abductive_example)
    examples_to_data_lists(anli_dataset.dev_examples, f'anli', 'dev', form_func=form_abductive_example)
    examples_to_data_lists(anli_dataset.test_examples, f'anli', 'test', form_func=form_abductive_example)


if __name__ == '__main__':
    # write_training_data_defeasible()
    write_training_data_abductive()
