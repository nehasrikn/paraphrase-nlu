import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Tuple, Union
import pandas as pd
import json
import random
import string
import os
import termplotlib as tpl
import socket

PROJECT_ROOT_DIR = '/fs/clip-projects/rlab/nehasrik/paraphrase-nlu'

def plot_and_save_countplot(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.countplot(data=pd.DataFrame(values, columns=['value']), x="value")
    plt.xticks(rotation=45, ha='right')
    plot.get_figure().savefig(fig_file)

def plot_and_save_hist(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.histplot(data=pd.DataFrame(values, columns=['value']), x="value", kde=True)
    plot.get_figure().savefig(fig_file)

def clean_paraphrase(p: str) -> str:
    """
    Lower case, remove surrounding whitespace, remove punctuation 
    """
    return p.strip().lower().translate(str.maketrans('', '', string.punctuation))

def load_jsonlines(path: str) -> List[Any]:
    if not path.startswith(PROJECT_ROOT_DIR):
        path = os.path.join(PROJECT_ROOT_DIR, path)

    with open(path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

def write_jsonlines(l: List[Any], path: str) -> None:
    if not path.startswith(PROJECT_ROOT_DIR):
        path = os.path.join(PROJECT_ROOT_DIR, path)

    with open(path, 'w') as f:
        for entry in l:
            json.dump(entry, f)
            f.write('\n')

def write_json(d: dict, path: str) -> None:
    if not path.startswith(PROJECT_ROOT_DIR):
        path = os.path.join(PROJECT_ROOT_DIR, path)

    with open(path, 'w') as fp:
        json.dump(d, fp)

def load_json(path: str) -> dict:
    if not path.startswith(PROJECT_ROOT_DIR):
        path = os.path.join(PROJECT_ROOT_DIR, path)

    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_example_kv_pair_from_dict(d) -> Tuple[Any, Any]:
    sample_k = random.sample(d.keys(), 1)[0]
    sample_v = d[sample_k]
    return sample_k, sample_v

def get_example_kv_pair_from_list_of_dicts(d):
    """
    Sample a random k,v pair from a list of dicts. Just to see the shape of the data.
    Example: [{'a': 1}, {'b': 2}, {'c': 3, 'd': 4}] ->  'd': 4
    """
    sample_dict = random.sample(d, 1)[0]
    sample_k, sample_v = get_example_kv_pair_from_dict(sample_dict)
    print(f'\nSample k,v pair from list of dicts: [{sample_k}: {sample_v}]')


def termplot(data: List[Union[int, float]]):
    counts, bin_edges = np.histogram(data, bins=10)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()

def write_strings_to_file(strings, path):
    if not path.startswith(PROJECT_ROOT_DIR):
        path = os.path.join(PROJECT_ROOT_DIR, path)

    with open(path, 'w') as f:
        for string in strings:
            f.write(str(string) + '\n')