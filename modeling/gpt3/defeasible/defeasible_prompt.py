from defeasible_data import (
    DefeasibleNLIExample,
    ParaphrasedDefeasibleNLIExample,
    dnli_datasets
)
import os
from typing import List, Optional, Dict, Union
import random
from utils import PROJECT_ROOT_DIR, load_json
from string import Template

DEFEASIBLE_INSTRUCTIONS = """Given a Scenario, an Evidence sentence is called a Weakener if the Scenario is less plausible after learning the Evidence sentence. Similarly, an Evidence sentence is called a Strengthener if the Scenario is more likely to be true after learning the Evidence sentence. Given a Scenario and Evidence, choose whether the Evidence sentence is a Strengthener (S) or a Weakener (W) using the following format:\n"""

INFERENCE_EXAMPLE_TEMPLATE = "Scenario: {premise} {hypothesis}\nEvidence: {update}\nAnswer:"

def add_period(string):
  if string and string[-1] not in [".", "!", "?"]:
    string += "."
  return string


def dnli_example_2_str(example: DefeasibleNLIExample) -> str:
    template = f"Scenario: {add_period(example.premise)} {add_period(example.hypothesis)}\nEvidence: {add_period(example.update)}\nAnswer: {example.update_type[0].upper()}"
    return template

def select_in_context_examples(num_examples_per_dataset: int) -> List[DefeasibleNLIExample]:
    """
    Selects num_examples_per_dataset examples from each dataset and returns them in a list.
    """
    icl_examples = []
    for dname, d in dnli_datasets.items():
        analysis_model_examples = load_json(os.path.join(PROJECT_ROOT_DIR, f'data_selection/defeasible/{dname}/analysis_model_examples/train_examples.json'))

        random.seed(2022)
        examples = random.sample(analysis_model_examples, num_examples_per_dataset)
        
        text = list(map(dnli_example_2_str, [DefeasibleNLIExample(**e) for e in examples]))
        icl_examples.extend(text)
    
    random.seed(42)
    return random.sample(icl_examples, len(icl_examples)) # shuffle

def construct_prompt_template(num_examples_per_dataset: int) -> str:
    """
    Constructs a prompt for the defeasible task with 
    num_examples_per_dataset examples from each dataset.
    and instructions.
    """
    icl_examples = ["#########\n" + e + "\n" for e in select_in_context_examples(num_examples_per_dataset)]
    instructions = DEFEASIBLE_INSTRUCTIONS + "\n" + "\n".join(icl_examples)
    return instructions + "\n#########\n" + INFERENCE_EXAMPLE_TEMPLATE

def form_prompt_with_example(prompt_template: str, example: Union[DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample]) -> str:
    """
    Forms a prompt for the defeasible task with the given example.
    """
    if isinstance(example, DefeasibleNLIExample):
        return prompt_template.format(
            premise=add_period(example.premise), 
            hypothesis=add_period(example.hypothesis), 
            update=add_period(example.update)
        )
    elif isinstance(example, ParaphrasedDefeasibleNLIExample):
        return prompt_template.format(
            premise=add_period(example.original_example.premise), 
            hypothesis=add_period(example.original_example.hypothesis), 
            update=add_period(example.update_paraphrase)
        )
    else: 
        raise TypeError("example must be of type DefeasibleNLIExample or ParaphrasedDefeasibleNLIExample")