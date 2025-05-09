from abductive_data import (
    AbductiveNLIExample,
    ParaphrasedAbductiveNLIExample,
    anli_dataset
)
import os
from typing import List, Optional, Dict, Union
import random
from utils import PROJECT_ROOT_DIR, load_json
from string import Template

ABDUCTIVE_INSTRUCTIONS = """Given a story with a beginning and an end, select from two choices the middle sentence that is most plausible and best explains the beginning and end of the story. Here are a few examples:
"""

INFERENCE_EXAMPLE_TEMPLATE_WITH_ANSWER = "Beginning: {obs1}\nEnd: {obs2}\nSentence A: {hyp1}\nSentence B: {hyp2}\nPlausible Middle Sentence: {label}"

INFERENCE_EXAMPLE_TEMPLATE = "Beginning: {obs1}\nEnd: {obs2}\nSentence A: {hyp1}\nSentence B: {hyp2}\nPlausible Middle Sentence:"

def anli_example_2_str(example: AbductiveNLIExample) -> str:
    return INFERENCE_EXAMPLE_TEMPLATE_WITH_ANSWER.format(
        obs1=example.obs1, obs2=example.obs2, hyp1=example.hyp1, hyp2=example.hyp2, label='A' if example.label == 1 else 'B'
    )

def select_in_context_examples(num_examples: int=10) -> List[AbductiveNLIExample]:
    """
    Selects num_examples_per_dataset examples from each dataset and returns them in a list.
    """
    random.seed(2022)
    icl_examples = random.sample(anli_dataset.train_examples, num_examples)
    text = list(map(anli_example_2_str, icl_examples))
    random.seed(42)
    return random.sample(text, len(text)) # shuffle

def construct_prompt_template(num_examples: int=10) -> str:
    """
    Constructs a prompt for the abductive task with 
    num_examples_per_dataset examples from each dataset.
    and instructions.
    """
    icl_examples = ["#########\n" + e + '\n' for e in select_in_context_examples(num_examples)]
    instructions = ABDUCTIVE_INSTRUCTIONS + "\n" + "\n".join(icl_examples)
    return instructions + "\n#########\n" + INFERENCE_EXAMPLE_TEMPLATE

def form_prompt_with_example(prompt_template: str, example: Union[AbductiveNLIExample, ParaphrasedAbductiveNLIExample]) -> str:
    """
    Forms a prompt for the defeasible task with the given example.
    """
    if isinstance(example, AbductiveNLIExample):
        return prompt_template.format(
            obs1=example.obs1,
            obs2=example.obs2,
            hyp1=example.hyp1,
            hyp2=example.hyp2,
        )
    elif isinstance(example, ParaphrasedAbductiveNLIExample):
        return prompt_template.format(
            obs1=example.original_example.obs1,
            obs2=example.original_example.obs2,
            hyp1=example.hyp1_paraphrase,
            hyp2=example.hyp2_paraphrase,
        )
    else: 
        raise TypeError("example must be of type AbductiveNLIExample or ParaphrasedAbductiveNLIExample")