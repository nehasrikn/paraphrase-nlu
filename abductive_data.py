import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import os
import json
from tqdm import tqdm
from utils import PROJECT_ROOT_DIR
from simple_colors import magenta, green, red, yellow


@dataclass
class AbductiveNLIExample:
    example_id: str
    source_example_metadata: Optional[Dict] #story_id, etc
    obs1: str
    obs2: str
    hyp1: str
    hyp2: str
    label: Optional[int] #1 or 2 corresponding to the correct hypothesis (1 = hyp1, 2 = hyp2)
    modeling_label: Optional[int] # label - 1 (0 or 1) for modeling purposes
    annotated_paraphrases: List[Dict[str, List[str]]]

    def __str__(self):
        return f'Obs1: {self.obs1}\nObs2: {self.obs2}\nHyp1: {self.hyp1}\nHyp2: {self.hyp2}'

@dataclass
class ParaphrasedAbductiveNLIExample:
    paraphrase_id: str # # <example_id>.<UUID>.<Paraphrase_Num_hyp1>.<paraphrased_num_hyp2> for human, <example_id>.<system>.<identifiers> for generated
    original_example: AbductiveNLIExample
    original_example_id: str
    hyp1_paraphrase: str
    hyp2_paraphrase: str
    worker_id: Optional[str] = None #mturk worker id or system
    obs1_paraphrase: Optional[str] = None
    obs2_paraphrase: Optional[str] = None
    automatic_system_metadata: Optional[Dict[Any, Any]] = None # can contain system-specific metadata

    def __str__(self):
        return f'Hyp1:{self.hyp1_paraphrase}\nHyp2:{self.hyp2_paraphrase}'
    
    def display(self, display_original_example=False):
        if display_original_example:
            print(green(f'Original Example ({self.original_example.example_id})', ['bold', 'underlined']))
            print(green('  [Obs1]'), self.original_example.obs1)
            print(green('  [Obs2]'), self.original_example.obs2)
            print(green('  [Hyp1]'), self.original_example.hyp1)
            print(green('  [Hyp2]'), self.original_example.hyp2)
            print(green('  [Label]'), self.original_example.label)
            print(green('  [Source Metadata]'), self.original_example.source_example_metadata)

        print(magenta('Paraphrased Example', ['bold', 'underlined']))
        print(magenta('  [Paraphrased ID]'), self.paraphrase_id)
        print(magenta('  [Paraphrased Hyp1]'), self.hyp1_paraphrase)
        print(magenta('  [Paraphrased Hyp2]'), self.hyp2_paraphrase)
        if self.worker_id:
            print(magenta('  [Worker ID]'), self.worker_id)


class AbductiveNLIDataset:

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.train_examples = self.create_examples(data_split='train') 
        self.dev_examples = self.create_examples(data_split='dev') 
        self.test_examples = self.create_examples(data_split='test') 

        self.easy_examples = self.create_easy_examples()

        self.split_examples_by_id = {split: {e.example_id: e for e in self.get_split(split)} for split in ['train', 'dev', 'test']}
        self.easy_examples_by_id = {e.example_id: e for e in self.easy_examples}

    def create_examples(self, data_split: str) -> List[AbductiveNLIExample]:
        examples = []
        data_fname = os.path.join(PROJECT_ROOT_DIR, '%s/%s.jsonl' % (self.data_dir, data_split))
        label_fname = os.path.join(PROJECT_ROOT_DIR, '%s/%s-labels.lst' % (self.data_dir, data_split))
        
        with open(label_fname) as label_file:
            labels = [line.rstrip() for line in label_file]
        data = [json.loads(json_str) for json_str in list(open(data_fname, 'r'))]
        assert len(data) == len(labels)

        for i, (example, label) in enumerate(zip(data, labels)):
            abductive_example = AbductiveNLIExample(
                example_id='anli.%s.%d' % (data_split, i),
                source_example_metadata={'story_id': example['story_id']},
                obs1=example['obs1'],
                obs2=example['obs2'],
                hyp1=example['hyp1'],
                hyp2=example['hyp2'],
                label=int(label),
                modeling_label=int(label) - 1,
                annotated_paraphrases=None
            )
            examples.append(abductive_example)
        
        # print('Loaded %d nonempty %s examples' % (len(data), data_split))
        return examples
    
    def create_easy_examples(self) -> List[AbductiveNLIExample]:
        fname = os.path.join(PROJECT_ROOT_DIR, 'raw-data/anli/af_filtered_out/train_easy_annotations.jsonl')
        easy_examples = []
        for i, json_str in enumerate(list(open(fname, 'r'))):
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
        return easy_examples

    def get_split(self, split_name: str) -> List[AbductiveNLIExample]:
        if split_name == 'train':
            return self.train_examples
        elif split_name == 'dev':
            return self.dev_examples
        else:
            return self.test_examples

    def get_example_by_id(self, example_id) -> AbductiveNLIExample:
        if 'easy' in example_id:
            return self.easy_examples_by_id[example_id]

        _, split, ex_id = example_id.split('.') #anli.test.2619
        return self.split_examples_by_id[split][example_id]

anli_dataset = AbductiveNLIDataset(data_dir='raw-data/anli')